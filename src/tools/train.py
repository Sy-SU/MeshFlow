# coding: utf-8
"""
文件路径：src/tools/train.py
用途：训练 flow_match_tokens 模型（仅以 x1 + t 为条件，引导速度场），
     使用 Heun 单步作为训练目标；并在验证集上以 mean(||x2_pred - x2_gt||) 进行评估、
     早停保存，以及可配置的学习率调度（含 warmup）。
"""

import os
import sys
import json
import time
import argparse
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 允许从项目根目录导入 src.*
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---- 数据与模型导入 ----
from src.datasets.dataloader import get_dataloader
from src.models.flow_matcher import (
    ShapeTokenizer,
    FlowFieldWithTokens,
)

# ------------------ 实用工具 ------------------
def _get_lr(optimizer: optim.Optimizer) -> Optional[float]:
    for pg in optimizer.param_groups:
        return float(pg.get("lr", None))
    return None

def build_scheduler(optimizer: optim.Optimizer, args, steps_per_epoch: int):
    """
    返回 (warmup_sched, main_sched)
    warmup_sched: 按 epoch 线性 warmup（可选）
    main_sched:   主调度器；OneCycle 按 step，其它按 epoch；Plateau 需传 val_metric
    """
    import torch.optim.lr_scheduler as lrs

    warmup = None
    if args.warmup_epochs > 0:
        def _lr_lambda(epoch_idx):
            # epoch 从 0 开始计；第 warmup_epochs 个 epoch 末尾提升至 1.0
            return min(1.0, float(epoch_idx + 1) / float(args.warmup_epochs))
        warmup = lrs.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    sched = None
    sch = args.scheduler.lower()
    if sch == "none":
        sched = None
    elif sch == "cosine":
        sched = lrs.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=args.min_lr)
    elif sch == "cosine_warm_restarts":
        sched = lrs.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.restart_period, T_mult=args.restart_mult, eta_min=args.min_lr
        )
    elif sch == "step":
        sched = lrs.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif sch == "multistep":
        milestones = [int(m.strip()) for m in args.milestones.split(",") if m.strip() != ""]
        sched = lrs.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    elif sch == "plateau":
        sched = lrs.ReduceLROnPlateau(
            optimizer, mode="min", patience=args.plateau_patience, factor=args.plateau_factor,
            min_lr=args.min_lr, verbose=True
        )
    elif sch == "onecycle":
        total_steps = steps_per_epoch * args.tmax  # 把 tmax 当作 total_epochs 使用
        # 注意：OneCycle 需要 per-step step()
        sched = lrs.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=max(1, total_steps),
            pct_start=args.onecycle_pct_start,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=max(args.lr / max(args.min_lr, 1e-8), 1.0)
        )
    else:
        sched = None

    return warmup, sched


# ------------------ 从 faces 构造 (x1, x2) + 有效 mask ------------------
def edges_from_faces(
    faces_b_f_3_3: torch.Tensor,
    max_edges_per_mesh: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    faces_b_f_3_3: [B, F, 3, 3]（padding 面全 0）
    返回：x1, x2: [B,N,3]；mask: [B,N]
    """
    B = faces_b_f_3_3.shape[0]
    device = faces_b_f_3_3.device
    x1_list, x2_list = [], []

    for b in range(B):
        faces = faces_b_f_3_3[b]  # [F,3,3]
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
    MSE with mask support.
    a, b: [B, N, 3]
    mask: [B, N] or None
    """
    diff2 = (a - b) ** 2         # [B, N, 3]
    err = diff2.sum(dim=-1)      # [B, N]

    if mask is not None:
        err = err[mask]          # flatten only valid entries

    if err.numel() == 0:         # avoid nan when no valid entries
        return torch.tensor(0.0, device=a.device, dtype=a.dtype, requires_grad=a.requires_grad)

    return err.mean()


def directional_loss(dx_pred: torch.Tensor, dx_gt: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    eps = 1e-8
    dot = (dx_pred * dx_gt).sum(dim=-1)  # [B,N]
    denom = dx_pred.norm(dim=-1) * dx_gt.norm(dim=-1) + eps

    cos_sim = dot / denom  # [B,N]
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # 防止数值超范围

    loss = 1.0 - cos_sim  # [B,N]

    if mask is not None:
        loss = loss[mask]

    if loss.numel() == 0:  # 防止 mean 空 tensor
        return torch.tensor(0.0, device=dx_pred.device, dtype=dx_pred.dtype)

    return loss.mean()



# ------------------ 验证：重建 x2_pred 并计算 mean(||x2_pred - x2_gt||) ------------------
@torch.no_grad()
def reconstruct_batch(
    flow: FlowFieldWithTokens,
    tokenizer: ShapeTokenizer,
    x1: torch.Tensor,
    num_steps: int = 100
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
        k1 = flow(x, t0, cond_tokens, cond_global)           # [B,N,3]
        x_tilde = x + dt * k1
        t1 = torch.full((B,), (k + 1) * dt, device=device, dtype=dtype)
        k2 = flow(x_tilde, t1, cond_tokens, cond_global)     # [B,N,3]
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
    device = next(flow.parameters()).device
    total_dist = 0.0
    total_count = 0

    for batch in val_loader:
        faces = batch
        if isinstance(faces, list):
            faces = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(f, dtype=torch.float32) for f in faces],
                                                    batch_first=True)
        faces = faces.to(device).float()  # [B,F,3,3]

        x1, x2_gt, mask = edges_from_faces(faces, max_edges_per_mesh)      # [B,N,3], [B,N,3], [B,N]
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
    epoch: int,
    main_sched: Optional[object] = None
) -> float:
    device = next(flow.parameters()).device
    flow.train(); tokenizer.train()

    running_loss = 0.0
    step_count = 0

    pbar = tqdm(train_loader, total=len(train_loader),
                desc=f"[Train] epoch {epoch}", dynamic_ncols=True)
    for batch in pbar:
        faces = batch
        if isinstance(faces, list):
            faces = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(f, dtype=torch.float32) for f in faces],
                                                    batch_first=True)
        faces = faces.to(device).float()  # [B,F,3,3]

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
            mse_loss = loss
            dir_loss = None
            if args.dir_weight > 0.0:
                loss = loss + args.dir_weight * directional_loss(dx_pred, dx_gt, mask)
                dir_loss = directional_loss(dx_pred, dx_gt, mask)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(flow.parameters()) + list(tokenizer.parameters()), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # OneCycle 是 per-step 调度；其它策略不在这里 step
        if main_sched is not None and isinstance(main_sched, torch.optim.lr_scheduler.OneCycleLR):
            main_sched.step()

        running_loss += loss.item()
        step_count += 1

        cur_lr = _get_lr(optimizer)
        if cur_lr is not None:
            pbar.set_postfix(loss=f"{running_loss/step_count:.6f}", mse_loss = f"{mse_loss:.6f}", dir_loss = f"{dir_loss:.6f}", lr=f"{cur_lr:.2e}")
        else:
            pbar.set_postfix(loss=f"{running_loss/step_count:.6f}")

    return running_loss / max(step_count, 1)


def save_checkpoint(flow, tokenizer, optimizer, epoch, best_metric, outdir, tag: str = "latest",
                    early_state: Optional[Dict[str, Any]] = None,
                    sched_state: Optional[Dict[str, Any]] = None):
    os.makedirs(outdir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "best_metric": best_metric,
        "flow": flow.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if early_state is not None:
        ckpt["early_state"] = early_state  # {'no_improve_epochs': int, 'cooldown': int}
    if sched_state is not None:
        ckpt["scheduler"] = sched_state
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
    early_state = state.get("early_state", {"no_improve_epochs": 0, "cooldown": 0})
    sched_state = state.get("scheduler", None)
    print(f"[Load] resume from {path}, start_epoch={start_epoch}, best_metric={best_metric:.6f}")
    return start_epoch, best_metric, early_state, sched_state


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
    ap.add_argument("--train_dt", type=float, default=0.01, help="Heun 单步的 Δt（训练目标）")
    ap.add_argument("--dir_weight", type=float, default=0.1, help="方向项权重（1 - cos），设 0 关闭")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", help="启用混合精度训练")

    # 验证
    ap.add_argument("--val_steps", type=int, default=100, help="重建时 Heun 积分步数（验证）")

    # 设备 & 日志
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--ckpt_dir", type=str, default="outs/ckpts",
                    help="权重保存目录。若提供具体目录，将直接使用该目录（不再拼 run_name）。")
    ap.add_argument("--run_name", type=str, default="flow_tokens")

    # 断点续训
    ap.add_argument("--resume", type=str, default=None, help="加载 ckpt 路径并继续训练")

    # ========= 模型结构超参数 =========
    ap.add_argument("--d_model", type=int, default=256, help="Transformer 通道维度")
    ap.add_argument("--nhead", type=int, default=8, help="多头注意力的头数")   # 修正了原来的小typo
    ap.add_argument("--tok_num_tokens", type=int, default=32, help="ShapeTokenizer 的 token 数")
    ap.add_argument("--tok_depth", type=int, default=2, help="ShapeTokenizer 的 Transformer 层数")
    ap.add_argument("--flow_layers", type=int, default=4, help="FlowFieldWithTokens 的层数")
    ap.add_argument("--pe_num_frequencies", type=int, default=16, help="坐标傅里叶编码频率数")
    ap.add_argument("--pe_include_input", action="store_true",
                    help="位置编码是否包含原始坐标（默认 False；与旧默认一致需显式打开）")

    # ========= Early Stopping =========
    ap.add_argument("--early_patience", type=int, default=10,
                    help="早停耐心轮数（验证指标在该轮数内未显著提升就停止）")
    ap.add_argument("--early_min_delta", type=float, default=0.0,
                    help="认为“有提升”的最小改变量(绝对值)；val_metric < best - min_delta 才算提升")
    ap.add_argument("--early_min_epochs", type=int, default=5,
                    help="在达到该最小轮数前不触发早停")
    ap.add_argument("--early_cooldown", type=int, default=0,
                    help="触发提升后冷却若干轮，在冷却期内不累计耐心计数")

    # ========= LR Scheduler =========
    ap.add_argument("--scheduler", type=str, default="none",
                    choices=["none", "cosine", "cosine_warm_restarts", "step", "multistep", "plateau", "onecycle"],
                    help="学习率调度策略")
    ap.add_argument("--warmup_epochs", type=int, default=0, help="线性 warmup 的 epoch 数")
    ap.add_argument("--min_lr", type=float, default=1e-6, help="Cosine 系列 / Plateau 的最小 LR")
    ap.add_argument("--tmax", type=int, default=50, help="Cosine 的 T_max 或 OneCycle 的 total_epochs")
    ap.add_argument("--restart_period", type=int, default=10, help="CosineWarmRestarts 的 T_0")
    ap.add_argument("--restart_mult", type=float, default=2.0, help="CosineWarmRestarts 的 T_mult")
    ap.add_argument("--step_size", type=int, default=30, help="StepLR 的 step_size")
    ap.add_argument("--gamma", type=float, default=0.1, help="Step/MultiStepLR 的 gamma")
    ap.add_argument("--milestones", type=str, default="", help="MultiStepLR 里程碑（逗号分隔，如 60,80）")
    ap.add_argument("--plateau_patience", type=int, default=5, help="ReduceLROnPlateau 的耐心轮数")
    ap.add_argument("--plateau_factor", type=float, default=0.5, help="ReduceLROnPlateau 的下降因子")
    ap.add_argument("--onecycle_pct_start", type=float, default=0.3, help="OneCycleLR pct_start")

    # 随机种子
    ap.add_argument("--seed", type=int, default=42)

    return ap


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    # 随机性
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- DataLoader ----
    train_loader, val_loader, _ = get_dataloader(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"[Data] train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    # ---- Model ----
    # pe_dim = (include_input ? 3 : 0) + 3 * 2 * num_frequencies
    pe_dim = (3 if args.pe_include_input else 0) + 3 * 2 * args.pe_num_frequencies

    tokenizer = ShapeTokenizer(
        pe_dim=pe_dim,
        d_model=args.d_model,
        num_tokens=args.tok_num_tokens,
        nhead=args.nhead,
        depth=args.tok_depth
    ).to(device)

    flow = FlowFieldWithTokens(
        num_frequencies=args.pe_num_frequencies,
        include_input=args.pe_include_input,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.flow_layers
    ).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(tokenizer)
    print(flow)
    print(f"Total parameters of tokenizer: {count_parameters(tokenizer):,}")
    print(f"Total parameters of flow: {count_parameters(flow):,}")

    # ---- Optimizer ----
    optimizer = optim.AdamW(
        list(flow.parameters()) + list(tokenizer.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=args.amp)

    # ---- Checkpoint 目录 ----
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.ckpt_dir and os.path.exists(args.ckpt_dir):
        run_dir = args.ckpt_dir
    else:
        run_dir = os.path.join("outs", "ckpts", args.run_name)
        os.makedirs(run_dir, exist_ok=True)

    # 记录配置
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ---- Resume (可能带早停与scheduler状态) ----
    start_epoch = 1
    best_val = float("inf")
    early_state: Dict[str, Any] = {"no_improve_epochs": 0, "cooldown": 0}
    sched_state = None
    if args.resume is not None and os.path.isfile(args.resume):
        start_epoch, best_val, early_state, sched_state = load_checkpoint(
            flow, tokenizer, optimizer, args.resume, device
        )

    # ---- Scheduler ----
    steps_per_epoch = len(train_loader)
    warmup_sched, main_sched = build_scheduler(optimizer, args, steps_per_epoch)
    if sched_state is not None and main_sched is not None:
        try:
            main_sched.load_state_dict(sched_state)
            print("[Load] scheduler state loaded.")
        except Exception as e:
            print(f"[Load] scheduler state load failed: {e}")

    # ---- Train Loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(flow, tokenizer, train_loader, optimizer, scaler, args, epoch, main_sched)
        t1 = time.time()

        # 验证：mean(||x2_pred - x2_gt||)
        val_metric = validate(
            flow, tokenizer, val_loader,
            num_steps=args.val_steps,
            max_edges_per_mesh=args.max_edges_per_mesh,
            epoch=epoch
        )
        t2 = time.time()

        print(f"[Epoch {epoch}] train_loss={train_loss:.6f}  "
              f"val_mean_dist={val_metric:.6f}  "
              f"lr={_get_lr(optimizer):.2e}  "
              f"time(train)={t1-t0:.1f}s  time(val)={t2-t1:.1f}s")

        # ========== Early Stopping 更新 ==========
        improved = (val_metric < best_val - args.early_min_delta)

        if improved:
            best_val = val_metric
            early_state["no_improve_epochs"] = 0
            early_state["cooldown"] = args.early_cooldown
            save_checkpoint(
                flow, tokenizer, optimizer, epoch, best_val, run_dir, tag="best",
                early_state=early_state,
                sched_state=(main_sched.state_dict() if main_sched is not None else None)
            )
        else:
            # 未改进
            if early_state.get("cooldown", 0) > 0:
                early_state["cooldown"] -= 1  # 冷却期内不累计耐心
            else:
                early_state["no_improve_epochs"] = early_state.get("no_improve_epochs", 0) + 1

        # 始终保存 latest（含早停 & scheduler 状态）
        save_checkpoint(
            flow, tokenizer, optimizer, epoch, best_val, run_dir, tag="latest",
            early_state=early_state,
            sched_state=(main_sched.state_dict() if main_sched is not None else None)
        )

        # --- Scheduler step (按策略与时机) ---
        import torch.optim.lr_scheduler as lrs
        # 1) 先处理 warmup（按 epoch）
        if warmup_sched is not None and epoch <= args.warmup_epochs:
            warmup_sched.step()

        # 2) 主调度
        if main_sched is not None:
            if isinstance(main_sched, lrs.ReduceLROnPlateau):
                main_sched.step(val_metric)  # 需要验证指标
            elif isinstance(main_sched, lrs.OneCycleLR):
                # OneCycle 已在每个 step 调用了，这里不再 step
                pass
            else:
                # 其它策略按 epoch 更新
                main_sched.step()

        # 触发早停：满足最小轮数且“未改进轮数 >= patience”
        if (epoch >= args.early_min_epochs and
            early_state.get("no_improve_epochs", 0) >= args.early_patience):
            print(f"[EarlyStop] Stop at epoch {epoch}: "
                  f"no improvement for {early_state['no_improve_epochs']} epochs "
                  f"(best={best_val:.6f}, min_delta={args.early_min_delta})")
            # 记录一个标记文件，便于脚本检测
            with open(os.path.join(run_dir, "early_stop.json"), "w") as f:
                json.dump({
                    "stopped_epoch": epoch,
                    "best_val": best_val,
                    "no_improve_epochs": early_state["no_improve_epochs"],
                    "patience": args.early_patience,
                    "min_delta": args.early_min_delta
                }, f, indent=2, ensure_ascii=False)
            break

    print("[Done] 训练完成。")


if __name__ == "__main__":
    main()
