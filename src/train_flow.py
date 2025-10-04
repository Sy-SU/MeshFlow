# coding: utf-8
# file: src/train_flow.py
"""
基线训练：用两头 MLP 预测 v2 与 v3，先打通训练全流程。
后续替换成“裸的流匹配”时，只需改模型与损失。
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.tri_predictor import TriPredictorCFM

class TripletNPZDataset(Dataset):
    """从 outs/data 下的 npz 读取 v1,v2,v3（全部拼接）。"""
    def __init__(self, root: str, max_files=None):
        self.samples = []
        files = sorted(Path(root).rglob("*.npz"))
        if max_files is not None:
            files = files[:max_files]
        for p in files:
            data = np.load(str(p))
            if not all(k in data for k in ("v1","v2","v3")):
                continue
            v1, v2, v3 = data["v1"], data["v2"], data["v3"]
            # 逐行收集，避免一次性巨量内存（这里先简单拼接，够用）
            self.samples.append((v1, v2, v3))
        if not self.samples:
            raise RuntimeError(f"在 {root} 下未找到包含 v1/v2/v3 的 npz")

        self.v1 = torch.from_numpy(np.concatenate([s[0] for s in self.samples], axis=0)).float()
        self.v2 = torch.from_numpy(np.concatenate([s[1] for s in self.samples], axis=0)).float()
        self.v3 = torch.from_numpy(np.concatenate([s[2] for s in self.samples], axis=0)).float()

    def __len__(self): return self.v1.shape[0]
    def __getitem__(self, i):
        return self.v1[i], self.v2[i], self.v3[i]

def train_epoch(model, loader, opt, device, desc="Train", steps2_train: int = 50):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for v1, v2, v3 in pbar:
        v1, v2, v3 = v1.to(device), v2.to(device), v3.to(device)

        with torch.no_grad():
            x2_hat = model.sample_x2(v1, steps=steps2_train)

        # Flow #1: x2 | x1
        loss2 = model.cfm_loss(model.flow2, x_target=v2, cond=v1, device=device)

        # Flow #2: x3 | x1, x2   （训练时 teacher forcing：用 GT x2 作为条件）
        cond_3 = torch.cat([v1, x2_hat.detach()], dim=-1)   # 若你要用预测的 x2_hat 训练，把 v2 换为 x2_hat.detach()，但需要先采样一轮，较慢
        loss3 = model.cfm_loss(model.flow3, x_target=v3, cond=cond_3, device=device)

        loss = loss2 + loss3
        opt.zero_grad()
        loss.backward()
        opt.step()

        bs = v1.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(model, loader, device, desc="Val"):
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    s2 = s3 = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for v1, v2, v3 in pbar:
        v1, v2, v3 = v1.to(device), v2.to(device), v3.to(device)
        x2_hat, x3_hat = model.sample_chain(v1, steps2=50, steps3=50)
        s2_b = mse(x2_hat, v2).item()
        s3_b = mse(x3_hat, v3).item()
        bs = v1.size(0)

        s2 += s2_b
        s3 += s3_b
        n += bs

        pbar.set_postfix(v2_mse=f"{s2_b/bs:.4f}", v3_mse=f"{s3_b/bs:.4f}")
    return s2 / max(n, 1), s3 / max(n, 1)

def main():
    ap = argparse.ArgumentParser(description="Baseline: MLP for (v1→v2) and (v1,v2→v3)")
    ap.add_argument("--data-root", type=str, default="outs/data")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="outs/ckpts/tri_predictor.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # dataset
    full = TripletNPZDataset(args.data_root, max_files=args.max_files)
    N = len(full)
    idx = np.arange(N); np.random.shuffle(idx)
    k = int(N * (1.0 - args.val_ratio))
    train_idx, val_idx = idx[:k], idx[k:]

    sub = torch.utils.data.Subset(full, train_idx.tolist())
    val = torch.utils.data.Subset(full, val_idx.tolist())

    train_loader = DataLoader(sub, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader   = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, drop_last=False)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriPredictorCFM(hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    

    # train
    best = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, device)
        v2_mse, v3_mse = eval_epoch(model, val_loader, device)
        print(f"[Epoch {ep:02d}] train_loss={tr_loss:.6f} | val_mse_v2={v2_mse:.6f} val_mse_v3={v3_mse:.6f}")
        score = v2_mse + v3_mse
        if score < best:
            best = score
            torch.save({"model": model.state_dict(),
                        "args": vars(args)}, args.out)
            print(f"  ↳ saved to {args.out}")

if __name__ == "__main__":
    main()
