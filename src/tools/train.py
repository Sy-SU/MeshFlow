"""
文件路径：src/tools/train.py
用途：训练 flow_match_tokens 模型（仅以 x1 + t 为条件，引导速度场），
     使用 Heun 单步作为训练目标；并在验证集上以 mean(||x2_pred - x2_gt||) 进行评估与早停/保存。
"""

import os
import sys
import json
import math
import time
import argparse
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 允许从项目根目录导入 src.*
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---- 数据与模型导入 ----
from src.datasets.dataloader import get_dataloader  # 你之前实现的 DataLoader
from src.models.flow_matcher import (
    ShapeTokenizer,
    FlowFieldWithTokens,
    heun_step,            # 训练用 Heun 单步（需梯度）
    FourierPositionalEncoding3D,  # 仅用于确定 pe_dim
)

# ------------------ 从 faces 构造 (x1, x2) + 有效 mask ------------------
def edges_from_faces(
    faces_b_f_3_3: torch.Tensor,
    max_edges_per_mesh: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    faces_b_f_3_3: [B, F, 3, 3]（padding 面全 0）
    返回：
        x1, x2: [B, N, 3]（pad 到同长）
        mask:   [B, N]（True 表示有效边）
    """
    # print(f"faces_b_f_3_3: {faces_b_f_3_3.shape}")
    B = faces_b_f_3_3.shape[0]
    device = faces_b_f_3_3.device
    x1_list, x2_list = [], []

    for b in range(B):
        # print(f"face : {faces_b_f_3_3.shape}")
        faces = faces_b_f_3_3[b]  # [F,3,3]
        # print(f"faces: {faces.shape}")
        valid = faces.abs().sum(dim=(1, 2)) > 0
        faces = faces[valid]
        if faces.numel() == 0:
            x1_list.append(torch.zeros(1, 3, device=device))
            x2_list.append(torch.zeros(1, 3, device=device))
            continue
        v1 = faces[:, 0, :]
        v2 = faces[:, 1, :]
        select = torch.randperm(v1.shape[0], device=device)[: min(v1.shape[0], max_edges_per_mesh)]
        x1_list.append(v1[select])
        x2_list.append(v2[select])

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


# ------------------ 训练用损失（Masked MSE + 可选方向项） ------------------
def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    a,b: [B,N,3]; mask: [B,N] or None
    """
    diff2 = (a - b) ** 2  # [B,N,3]
    err = diff2.sum(dim=-1)  # [B,N]
    if mask is not None:
        err = err[mask]
    return err.mean()

def directional_loss(dx_pred: torch.Tensor, dx_gt: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    方向项：1 - cos(dx_pred, dx_gt)
    """
    eps = 1e-8
    dot = (dx_pred * dx_gt).sum(dim=-1)            # [B,N]
    denom = dx_pred.norm(dim=-1) * dx_gt.norm(dim=-1) + eps
    loss = 1.0 - (dot / denom)
    if mask is not None:
        loss = loss[mask]
    return loss.mean()


# ------------------ 验证：重建 x2_pred 并计算 mean(||x2_pred - x2_gt||) ------------------
@torch.no_grad()
def reconstruct_batch(
    flow: FlowFieldWithTokens,
    tokenizer: ShapeTokenizer,
    x1: torch.Tensor,
    num_steps: int = 20
) -> torch.Tensor:
    """
    Heun 多步从 x1 -> x2_pred；仅依赖 x1
    """
    flow.eval(); tokenizer.eval()
    B, N, _ = x1.shape
    device, dtype = x1.device, x1.dtype
    dt = 1.0 / float(num_steps)

    cond_tokens, cond_global = tokenizer(x1, flow.pe3d)  # [B,K,d], [B,d]
    x = x1.clone()
    for k in range(num_steps):
        t0 = torch.full((B,), k * dt, device=device, dtype=dtype)
        k1 = flow(x, t0, cond_tokens, cond_global)
        x_tilde = x + dt * k1
        t1 = torch.full((B,), (k + 1) * dt, device=device, dtype=dtype)
        k2 = flow(x_tilde, t1, cond_tokens, cond_global)
        x = x + 0.5 * (k1 + k2) * dt
    return x


@torch.no_grad()
def validate(
    flow: FlowFieldWithTokens,
    tokenizer: ShapeTokenizer,
    val_loader,
    num_steps: int,
    max_edges_per_mesh: int,
    epoch: int
) -> float:
    """
    返回：验证集平均重建误差 mean(||x2_pred - x2_gt||)
    """
    device = next(flow.parameters()).device
    total_dist = 0.0
    total_count = 0

    for batch in val_loader:
        # dataloader 返回 (vertices, faces)，我们只用 faces
        faces = batch
        if isinstance(faces, list):
            faces = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(f, dtype=torch.float32) for f in faces],
                                                    batch_first=True)  # 兜底
        faces = faces.to(device).float()  # [B,F,3,3]
        # print(f"face shape in validate: {faces.shape}")

        x1, x2_gt, mask = edges_from_faces(faces, max_edges_per_mesh)  # [B,N,3], [B,N,3], [B,N]
        x2_pred = reconstruct_batch(flow, tokenizer, x1, num_steps=num_steps)  # [B,N,3]

        dist = torch.norm(x2_pred - x2_gt, dim=-1)  # [B,N]
        valid = dist[mask]
        total_dist += valid.sum().item()
        total_count += valid.numel()

    mean_dist = total_dist / max(total_count, 1)
    return mean_dist


# ------------------ 训练主循环 ------------------
def train_one_epoch(
    flow: FlowFieldWithTokens,
    tokenizer: ShapeTokenizer,
    train_loader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    args,
    epoch: int
) -> float:
    """
    返回该 epoch 的平均训练损失
    """
    device = next(flow.parameters()).device
    flow.train(); tokenizer.train()

    running_loss = 0.0
    step_count = 0

    pbar = tqdm(train_loader, total=len(train_loader),
                desc=f"[Train] epoch {epoch}", dynamic_ncols=True)
    for batch in pbar:
        # print(f"batch shape: {batch.shape}")
        faces = batch
        if isinstance(faces, list):
            faces = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(f, dtype=torch.float32) for f in faces],
                                                    batch_first=True)
        faces = faces.to(device).float()  # [B,F,3,3]
        # print(f"face shape in train one epoch: {faces.shape}")

        # 取边监督
        x1, x2, mask = edges_from_faces(faces, args.max_edges_per_mesh)  # [B,N,3], [B,N,3], [B,N]

        # 采样 t, dt（保证 t+dt <= 1）
        B = x1.shape[0]
        dt = args.train_dt
        t = torch.rand(B, device=device, dtype=x1.dtype) * (1.0 - dt)

        # 构造监督（仅用于目标，不进入模型）
        v_gt = x2 - x1
        x_t = x1 + t[:, None, None] * v_gt
        dx_gt = dt * v_gt

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            # 条件仅依赖 x1（tokens 在该 batch 内共享，计算两次速度）
            cond_tokens, cond_global = tokenizer(x1, flow.pe3d)
            k1 = flow(x_t, t, cond_tokens, cond_global)           # [B,N,3]
            x_tilde = x_t + dt * k1
            t_next = t + dt
            k2 = flow(x_tilde, t_next, cond_tokens, cond_global)
            dx_pred = 0.5 * (k1 + k2) * dt

            # 损失（masked MSE + 可选方向项）
            loss = masked_mse(dx_pred, dx_gt, mask)
            if args.dir_weight > 0.0:
                loss = loss + args.dir_weight * directional_loss(dx_pred, dx_gt, mask)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(flow.parameters()) + list(tokenizer.parameters()), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        step_count += 1

        pbar.set_postfix(loss=f"{running_loss/step_count:.6f}")

    return running_loss / max(step_count, 1)


def save_checkpoint(flow, tokenizer, optimizer, epoch, best_metric, outdir, tag="latest"):
    os.makedirs(outdir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "best_metric": best_metric,
        "flow": flow.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = os.path.join(outdir, f"ckpt_{tag}.pt")
    torch.save(ckpt, path)
    print(f"[Save] checkpoint -> {path}")


def load_checkpoint(flow, tokenizer, optimizer, path, device):
    state = torch.load(path, map_location=device)
    flow.load_state_dict(state.get("flow", state), strict=False)
    tokenizer.load_state_dict(state.get("tokenizer", state), strict=False)
    if "optimizer" in state and optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    start_epoch = int(state.get("epoch", 0)) + 1
    best_metric = float(state.get("best_metric", float("inf")))
    print(f"[Load] resume from {path}, start_epoch={start_epoch}, best_metric={best_metric:.6f}")
    return start_epoch, best_metric


# ------------------ CLI ------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Train flow_match_tokens with Heun objective")
    # 数据
    ap.add_argument("--data_root", type=str, default="outs/data", help="数据根目录（包含 train/val/test）")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_edges_per_mesh", type=int, default=2048)

    # 训练
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--train_dt", type=float, default=0.05, help="Heun 单步的 Δt（训练目标）")
    ap.add_argument("--dir_weight", type=float, default=0.1, help="方向项权重（1 - cos），设 0 关闭")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", help="启用混合精度训练")

    # 验证
    ap.add_argument("--val_steps", type=int, default=20, help="重建时 Heun 积分步数（验证）")

    # 设备 & 日志
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--ckpt_dir", type=str, default="outs/ckpts")
    ap.add_argument("--run_name", type=str, default="flow_tokens")

    # 断点续训
    ap.add_argument("--resume", type=str, default=None, help="加载 ckpt 路径并继续训练")
    return ap


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---- DataLoader ----
    train_loader, val_loader, _ = get_dataloader(args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"[Data] train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    # ---- Model ----
    d_model = 256
    pe_dim = 3 + 3 * 2 * 16  # 对应 FlowFieldWithTokens 默认 num_frequencies=16, include_input=True
    tokenizer = ShapeTokenizer(pe_dim=pe_dim, d_model=d_model, num_tokens=32, nhead=8, depth=2).to(device)
    flow = FlowFieldWithTokens(num_frequencies=16, include_input=True, d_model=d_model, nhead=8, layers=4).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(tokenizer)  # 先打印结构
    print(flow)
    print(f"Total parameters of tokenizer: {count_parameters(tokenizer):,}")
    print(f"Total parameters of flow: {count_parameters(flow):,}")

    # ---- Optimizer ----
    optimizer = optim.AdamW(list(flow.parameters()) + list(tokenizer.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    # ---- Resume ----
    start_epoch = 1
    best_val = float("inf")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    run_dir = os.path.join(args.ckpt_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if args.resume is not None and os.path.isfile(args.resume):
        start_epoch, best_val = load_checkpoint(flow, tokenizer, optimizer, args.resume, device)

    # ---- Train Loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(flow, tokenizer, train_loader, optimizer, scaler, args, epoch)
        t1 = time.time()

        # 验证：mean(||x2_pred - x2_gt||)
        val_metric = validate(flow, tokenizer, val_loader, num_steps=args.val_steps,
                              max_edges_per_mesh=args.max_edges_per_mesh, epoch=epoch)
        t2 = time.time()

        print(f"[Epoch {epoch}] train_loss={train_loss:.6f}  "
              f"val_mean_dist={val_metric:.6f}  "
              f"time(train)={t1-t0:.1f}s  time(val)={t2-t1:.1f}s")

        # 保存最新
        save_checkpoint(flow, tokenizer, optimizer, epoch, best_val, run_dir, tag="latest")

        # 保存最佳（按 val_mean_dist 越小越好）
        if val_metric < best_val:
            best_val = val_metric
            save_checkpoint(flow, tokenizer, optimizer, epoch, best_val, run_dir, tag="best")

    print("[Done] 训练完成。")


if __name__ == "__main__":
    main()
