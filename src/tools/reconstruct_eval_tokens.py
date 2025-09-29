"""
文件路径：src/tools/reconstruct_eval_tokens.py
用途：加载训练得到的 Tokenizer 与流场模型权重，仅用 x1 与时间 t 通过 Heun 多步积分重建 x2_pred，
     并计算重建精度（平均欧氏距离）：mean(||x2_pred - x2_gt||)。若未提供权重将直接报错。
"""

import os
import argparse
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import sys

# 允许从项目根目录导入 src.*
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# === 引入你现有的模型实现（与 flow_matcher.py 对齐） ===
from src.models.flow_matcher import (
    ShapeTokenizer,
    FlowFieldWithTokens,
)

# ------------------ 数据辅助：从 faces 构造 (x1, x2) + 有效 mask ------------------
def edges_from_faces(
    faces_b_f_3_3: torch.Tensor,
    max_edges_per_mesh: int = 4096
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, num_faces, _, _ = faces_b_f_3_3.shape
    device = faces_b_f_3_3.device
    x1_list, x2_list = [], []

    for b in range(B):
        faces = faces_b_f_3_3[b]                          # [F,3,3]
        valid = faces.abs().sum(dim=(1, 2)) > 0           # 过滤 padding 面
        faces = faces[valid]
        if faces.numel() == 0:
            x1_list.append(torch.zeros(1, 3, device=device))
            x2_list.append(torch.zeros(1, 3, device=device))
            continue

        v1 = faces[:, 0, :]
        v2 = faces[:, 1, :]
        idx = torch.randperm(v1.shape[0], device=device)[:min(v1.shape[0], max_edges_per_mesh)]
        x1_list.append(v1[idx])
        x2_list.append(v2[idx])

    max_len = max(x.shape[0] for x in x1_list)

    def pad_to(x, L):
        if x.shape[0] == L:
            return x
        pad = torch.zeros(L - x.shape[0], 3, device=device, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    x1 = torch.stack([pad_to(x, max_len) for x in x1_list], dim=0)  # [B,N,3]
    x2 = torch.stack([pad_to(x, max_len) for x in x2_list], dim=0)  # [B,N,3]
    mask = (x1.abs().sum(dim=-1) > 0) & (x2.abs().sum(dim=-1) > 0)  # [B,N]
    return x1, x2, mask


# ------------------ 重建器：Heun 多步积分（只用 x1 作为条件） ------------------
class HeunReconstructor(nn.Module):
    def __init__(self, flow: FlowFieldWithTokens, tokenizer: ShapeTokenizer, num_steps: int = 20):
        super().__init__()
        self.flow = flow.eval()
        self.tokenizer = tokenizer.eval()
        self.num_steps = num_steps

    @torch.no_grad()
    def reconstruct(self, x1: torch.Tensor) -> torch.Tensor:
        B, N, _ = x1.shape
        device, dtype = x1.device, x1.dtype
        dt = 1.0 / float(self.num_steps)

        # 条件 tokens 仅依赖 x1，整段积分期间固定
        cond_tokens, cond_global = self.tokenizer(x1, self.flow.pe3d)  # [B,K,d], [B,d]

        x = x1.clone()
        for k in range(self.num_steps):
            t0 = torch.full((B,), k * dt, device=device, dtype=dtype)
            k1 = self.flow(x, t0, cond_tokens, cond_global)               # [B,N,3]
            x_tilde = x + dt * k1
            t1 = torch.full((B,), (k + 1) * dt, device=device, dtype=dtype)
            k2 = self.flow(x_tilde, t1, cond_tokens, cond_global)
            x = x + 0.5 * (k1 + k2) * dt
        return x  # x2_pred


# ------------------ 评估指标：平均欧氏距离 mean(||x2_pred - x2_gt||) ------------------
def mean_reconstruction_error(
    x2_pred: torch.Tensor,
    x2_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    d = torch.norm(x2_pred - x2_gt, dim=-1)  # [B,N]
    if mask is not None:
        d = d[mask]
    return float(d.mean().item())


# ------------------ I/O：读取 npz 中的 faces ------------------
def load_faces_from_npz(npz_path: str) -> torch.Tensor:
    arr = np.load(npz_path, allow_pickle=True)
    if 'faces' not in arr:
        raise KeyError(f"{npz_path} 缺少 'faces' 键")
    faces = arr['faces']  # (F,3,3)
    faces = torch.tensor(faces, dtype=torch.float32).unsqueeze(0)  # [1,F,3,3]
    return faces


# ------------------ 权重加载工具 ------------------
def _strip_module(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """去掉 DataParallel/DDP 的 'module.' 前缀"""
    if not state_dict:
        return state_dict
    if next(iter(state_dict)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def load_weights(
    flow: FlowFieldWithTokens,
    tokenizer: ShapeTokenizer,
    flow_ckpt: Optional[str],
    tokenizer_ckpt: Optional[str],
    joint_ckpt: Optional[str],
    device: torch.device
):
    if joint_ckpt:
        if not os.path.isfile(joint_ckpt):
            raise FileNotFoundError(joint_ckpt)
        state = torch.load(joint_ckpt, map_location=device)
        # 期望 {"flow": state_dict, "tokenizer": state_dict}
        if not isinstance(state, dict):
            raise ValueError("joint_ckpt 不是 dict，无法解析")
        if "flow" in state and "tokenizer" in state:
            flow_sd = _strip_module(state["flow"])
            tok_sd = _strip_module(state["tokenizer"])
            flow.load_state_dict(flow_sd, strict=False)
            tokenizer.load_state_dict(tok_sd, strict=False)
        else:
            # 兜底：如果直接是整个模型的 state_dict，也尝试硬加载（可能失败）
            flow.load_state_dict(_strip_module(state), strict=False)
            # tokenizer 可能在另一个文件：若未提供则报错
            if tokenizer_ckpt is None:
                raise ValueError("joint_ckpt 不含 'tokenizer'，请额外提供 --tokenizer_ckpt")
    else:
        # 分开加载
        if flow_ckpt is None or tokenizer_ckpt is None:
            raise ValueError("请提供 --flow_ckpt 与 --tokenizer_ckpt，或改用 --joint_ckpt")
        if not os.path.isfile(flow_ckpt) or not os.path.isfile(tokenizer_ckpt):
            raise FileNotFoundError("flow_ckpt 或 tokenizer_ckpt 路径不存在")

        flow_sd = _strip_module(torch.load(flow_ckpt, map_location=device))
        tok_sd  = _strip_module(torch.load(tokenizer_ckpt, map_location=device))
        flow.load_state_dict(flow_sd, strict=False)
        tokenizer.load_state_dict(tok_sd, strict=False)

    flow.eval()
    tokenizer.eval()


# ------------------ 评估入口：支持文件或目录 ------------------
def evaluate_path(
    path: str,
    flow: FlowFieldWithTokens,
    tokenizer: ShapeTokenizer,
    num_steps: int = 20,
    max_edges_per_mesh: int = 4096
):
    device = next(flow.parameters()).device
    reconstructor = HeunReconstructor(flow, tokenizer, num_steps=num_steps).to(device)

    def eval_one(npz_path: str) -> float:
        faces = load_faces_from_npz(npz_path).to(device)              # [1,F,3,3]
        x1, x2_gt, mask = edges_from_faces(faces, max_edges_per_mesh) # [1,N,3], [1,N,3], [1,N]
        x2_pred = reconstructor.reconstruct(x1)                       # [1,N,3]
        return mean_reconstruction_error(x2_pred, x2_gt, mask=mask)

    if os.path.isdir(path):
        npz_files: List[str] = [
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npz")
        ]
        if not npz_files:
            raise FileNotFoundError(f"{path} 下没有 .npz 文件")
        scores = []
        for f in npz_files:
            m = eval_one(f)
            print(f"[{os.path.basename(f)}] mean per face ||x2_pred - x2_gt|| = {m:.6f}")
            scores.append(m)
        overall = float(np.mean(scores)) if scores else float("nan")
        print(f"\n[OVERALL] mean per face over {len(scores)} files = {overall:.6f}")
        return overall
    else:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        m = eval_one(path)
        print(f"[{os.path.basename(path)}] mean per face ||x2_pred - x2_gt|| = {m:.6f}")
        return m


# ------------------ CLI ------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Heun 重建评估（需加载已训练权重）")
    ap.add_argument("--data", type=str, required=True, help="待评估的 .npz 文件或目录（outs/data/val/...）")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=20, help="Heun 积分步数")
    ap.add_argument("--max_edges", type=int, default=4096, help="每个样本最多使用多少条边")

    # 权重：二选一（分开 or 合并）
    ap.add_argument("--flow_ckpt", type=str, default=None, help="流场模型权重路径（分开保存时使用）")
    ap.add_argument("--tokenizer_ckpt", type=str, default=None, help="Tokenizer 权重路径（分开保存时使用）")
    ap.add_argument("--joint_ckpt", type=str, default=None, help="合并权重路径（包含 flow 与 tokenizer）")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    # 实例化模型（结构需与训练时一致）
    d_model = 256
    tokenizer = ShapeTokenizer(
        pe_dim=(3 + 3 * 2 * 16),  # 与 FlowFieldWithTokens 的默认 PE 设置对齐（num_frequencies=16, include_input=True）
        d_model=d_model, num_tokens=32, nhead=8, depth=2
    ).to(device)
    flow = FlowFieldWithTokens(
        num_frequencies=16, include_input=True,
        d_model=d_model, nhead=8, layers=4
    ).to(device)

    # 必须加载权重
    load_weights(
        flow, tokenizer,
        flow_ckpt=args.flow_ckpt,
        tokenizer_ckpt=args.tokenizer_ckpt,
        joint_ckpt=args.joint_ckpt,
        device=device
    )

    # 评估
    evaluate_path(args.data, flow, tokenizer, num_steps=args.steps, max_edges_per_mesh=args.max_edges)
