"""
文件路径：src/models/flow_match_tokens.py
用途：在“仅使用 x1 + t 条件引导”的前提下，参考 3D-Shape-Tokenization 的做法，
     将条件编码器升级为 Tokenizer（从 x1 产生 K 个形状 token），
     并在流场 Transformer 中通过跨注意力与 AdaLN/FiLM 注入条件，配合 Heun 单步训练。
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 位置/时间编码，沿用你现有实现 ----------
class FourierPositionalEncoding3D(nn.Module):
    def __init__(self, num_frequencies=32, include_input=True, log_scale=True):
        super().__init__()
        self.include_input = include_input
        if log_scale:
            self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            self.freq_bands = torch.linspace(1.0, 2.0 ** (num_frequencies - 1), num_frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N, 3]
        assert coords.ndim == 2 and coords.shape[1] == 3
        coords = coords.unsqueeze(-1)  # [N, 3, 1]
        freq = self.freq_bands.to(coords.device).reshape(1, 1, -1) * math.pi  # [1,1,L]
        scaled = coords * freq
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        enc = torch.cat([sin_feat, cos_feat], dim=-1).view(coords.shape[0], -1)  # [N, 3*2L]
        if self.include_input:
            enc = torch.cat([coords.squeeze(-1), enc], dim=-1)  # [N, 3 + 3*2L]
        return enc


class TimeEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.freqs = 2 * torch.pi * torch.logspace(0, 16, steps=16, base=2.0)
        self.fc1 = nn.Linear(32, 64)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(64, d)

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        freqs = self.freqs.to(t.device)[None, :]  # [1,16]
        emb = torch.cat([torch.sin(freqs * t), torch.cos(freqs * t)], dim=-1)  # [B,32]
        return self.fc2(self.act(self.fc1(emb)))  # [B,d]


def fourier_encode_batch(points_bnc: torch.Tensor,
                         pe: FourierPositionalEncoding3D) -> torch.Tensor:
    # points_bnc: [B,N,3] -> [B,N,Cpe]
    B, N, _ = points_bnc.shape
    flat = points_bnc.reshape(B * N, 3)
    enc = pe(flat).reshape(B, N, -1)
    return enc


# ---------- 模型组件：AdaLN（条件化层归一化） ----------
class AdaLN(nn.Module):
    """
    将条件向量 cond_vec 映射到 (scale, shift)，对输入做仿射调制。
    """
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 2 * d_model)
        )

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor):
        """
        x: [B, N, d_model]
        cond_vec: [B, d_cond] 或 [B, N, d_cond]（会自动广播）
        """
        x_norm = self.norm(x)
        if cond_vec.ndim == 2:
            cond_vec = cond_vec[:, None, :]  # [B,1,d_cond] 逐 token 广播
        ss = self.to_scale_shift(cond_vec)  # [B,N,2*d_model]
        scale, shift = ss.chunk(2, dim=-1)
        return x_norm * (1 + scale) + shift


# ---------- 模型组件：跨注意力（状态 <-- 条件tokens） ----------
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_mult: float = 4.0):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * ff_mult)),
            nn.GELU(),
            nn.Linear(int(d_model * ff_mult), d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x_state: torch.Tensor, cond_tokens: torch.Tensor):
        """
        x_state:    [B, Ns, d]  （查询）
        cond_tokens:[B, K,  d]  （键值）
        """
        q = self.q_proj(self.ln1(x_state))
        k = self.k_proj(cond_tokens)
        v = self.v_proj(cond_tokens)
        x = x_state + self.attn(q, k, v, need_weights=False)[0]
        x = x + self.ff(self.ln2(x))
        return x


# ---------- 条件编码器：Tokenizer（从 x1 → K 个 tokens） ----------
class ShapeTokenizer(nn.Module):
    """
    用 learnable queries 对 x1 的点/边集合做跨注意力聚合，产出 K 个 shape tokens。
    """
    def __init__(self, pe_dim: int, d_model: int, num_tokens: int = 32, nhead: int = 8, depth: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(pe_dim, d_model, bias=False)
        self.token_queries = nn.Parameter(torch.randn(num_tokens, d_model))
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead) for _ in range(depth)
        ])
        self.post = nn.LayerNorm(d_model)

    def forward(self, x1_b_n3: torch.Tensor, pe: FourierPositionalEncoding3D) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x1_b_n3: [B,N,3]（所有监督边的起点，或从 faces 派生的一批点）
        返回：
          cond_tokens: [B,K,d_model]
          cond_global: [B,d_model]（tokens 的均值，可做 AdaLN 条件向量）
        """
        B, N, _ = x1_b_n3.shape
        pe_x1 = fourier_encode_batch(x1_b_n3, pe)     # [B,N,Cpe]
        x = self.in_proj(pe_x1)                        # [B,N,d]
        # 广播 learnable queries 为 batch 维
        queries = self.token_queries[None, :, :].expand(B, -1, -1)  # [B,K,d]
        tokens = queries
        for blk in self.blocks:
            tokens = blk(tokens, x)                    # tokens 作为 query，x 作为 key/value
        tokens = self.post(tokens)                     # [B,K,d]
        cond_global = tokens.mean(dim=1)               # [B,d]
        return tokens, cond_global


# ---------- 流场 Transformer（加入跨注意力 + AdaLN 注入） ----------
class FlowFieldWithTokens(nn.Module):
    """
    fθ(x_state, t | tokens(x1))
    - 状态串：x_state 经 PE 和线性投影 -> [B,Ns,d]
    - 层结构： [SelfAttn -> FF -> CrossAttn(tokens) -> FF] × L，每层都用 AdaLN(cond_global)
    """
    def __init__(self,
                 num_frequencies: int = 16,
                 include_input: bool = True,
                 d_model: int = 256,
                 nhead: int = 8,
                 layers: int = 4,
                 ff_mult: float = 4.0):
        super().__init__()
        self.pe3d = FourierPositionalEncoding3D(num_frequencies, include_input, log_scale=True)
        pe_dim = 3 + 3 * 2 * num_frequencies if include_input else 3 * 2 * num_frequencies

        self.in_proj = nn.Linear(pe_dim, d_model, bias=False)
        self.time_enc = TimeEncoder(d_model)

        self.self_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=int(d_model * ff_mult),
                activation="gelu", batch_first=True, norm_first=True
            ) for _ in range(layers)
        ])
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, ff_mult) for _ in range(layers)
        ])
        self.adalns_self = nn.ModuleList([AdaLN(d_model, d_model) for _ in range(layers)])
        self.adalns_cross = nn.ModuleList([AdaLN(d_model, d_model) for _ in range(layers)])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3)
        )

    def forward(self, x_state: torch.Tensor, t: torch.Tensor,
                cond_tokens: torch.Tensor, cond_global: torch.Tensor) -> torch.Tensor:
        """
        x_state:     [B,Ns,3]
        t:           [B] / [B,1]
        cond_tokens: [B,K,d]
        cond_global: [B,d]
        return: u_pred [B,Ns,3]
        """
        B, Ns, _ = x_state.shape
        if t.ndim == 1:
            t = t[:, None]

        pe_x = fourier_encode_batch(x_state, self.pe3d)   # [B,Ns,Cpe]
        tok = self.in_proj(pe_x)                          # [B,Ns,d]

        t_emb = self.time_enc(t.squeeze(-1))[:, None, :].expand(B, Ns, -1)
        tok = tok + t_emb                                 # 时间注入（加法基线）

        # 交替：自注意力 + AdaLN(cond_global) + 跨注意力(tokens) + AdaLN
        for i, (blk_s, blk_c, adaln_s, adaln_c) in enumerate(
            zip(self.self_blocks, self.cross_blocks, self.adalns_self, self.adalns_cross)
        ):
            tok = adaln_s(tok, cond_global)
            tok = blk_s(tok)                   # self-attn
            tok = adaln_c(tok, cond_global)
            tok = blk_c(tok, cond_tokens)      # cross-attn (query=state, kv=cond tokens)

        return self.head(tok)                  # [B,Ns,3]


# ---------- Heun 单步（训练用），以及监督构造 ----------
def gt_path_and_step(x1: torch.Tensor, x2: torch.Tensor, t: torch.Tensor, dt: float):
    v_gt = x2 - x1
    x_t = x1 + t[:, None, None] * v_gt
    dx_gt = dt * v_gt
    return x_t, dx_gt

def heun_step(model: FlowFieldWithTokens, x_t, t, x1_for_tokens, tokenizer: ShapeTokenizer, dt: float):
    # 条件仅由 x1 构造，严格不看 x2
    cond_tokens, cond_global = tokenizer(x1_for_tokens, model.pe3d)  # [B,K,d], [B,d]
    k1 = model(x_t, t, cond_tokens, cond_global)                     # [B,N,3]
    x_tilde = x_t + dt * k1
    t_next = t + dt
    k2 = model(x_tilde, t_next, cond_tokens, cond_global)
    dx_pred = 0.5 * (k1 + k2) * dt
    return dx_pred

def heun_step_loss(model: FlowFieldWithTokens, tokenizer: ShapeTokenizer,
                   x1: torch.Tensor, x2: torch.Tensor, t: torch.Tensor, dt: float,
                   cos_w: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    loss = MSE(Δx_pred, Δx_gt) + cos_w * (1 - cos_sim)
    其中 Δx_gt = dt * (x2-x1)，模型输入不含 x2。
    """
    x_t, dx_gt = gt_path_and_step(x1, x2, t, dt)
    dx_pred = heun_step(model, x_t, t, x1, tokenizer, dt)
    mse = F.mse_loss(dx_pred, dx_gt)
    if cos_w > 0:
        # 方向项：1 - cos(Δx_pred, Δx_gt)
        eps = 1e-8
        num = (dx_pred * dx_gt).sum(-1)
        den = dx_pred.norm(dim=-1) * dx_gt.norm(dim=-1) + eps
        cos = (num / den).mean()
        loss = mse + cos_w * (1.0 - cos)
    else:
        loss = mse
    return loss, dx_pred, dx_gt


# ---------- 从 faces 构边（v1->v2），与之前一致 ----------
def edges_from_faces(faces_b_f_3_3: torch.Tensor, max_edges_per_mesh: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    B, n_faces, _, _ = faces_b_f_3_3.shape
    device = faces_b_f_3_3.device
    x1_list, x2_list = [], []
    for b in range(B):
        faces = faces_b_f_3_3[b]                               # [F,3,3]
        valid = faces.abs().sum(dim=(1, 2)) > 0
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
        if x.shape[0] == L: return x
        pad = torch.zeros(L - x.shape[0], 3, device=device, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    x1 = torch.stack([pad_to(x, max_len) for x in x1_list], dim=0)
    x2 = torch.stack([pad_to(x, max_len) for x in x2_list], dim=0)
    return x1, x2


# ---------- 迷你自检 ----------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, num_faces = 2, 1200
    faces = torch.randn(B, num_faces, 3, 3, device=device)

    # 取边监督 (v1->v2)
    x1, x2 = edges_from_faces(faces, max_edges_per_mesh=1024)  # [B,N,3]
    B = x1.shape[0]
    dt = 0.05
    t = torch.rand(B, device=device) * (1.0 - dt)

    # 实例化：Tokenizer + 流场
    d_model = 256
    tokenizer = ShapeTokenizer(
        pe_dim=(3 + 3 * 2 * 16),  # 与 FourierPositionalEncoding3D(num_frequencies=16, include_input=True) 对齐
        d_model=d_model, num_tokens=32, nhead=8, depth=2
    ).to(device)

    model = FlowFieldWithTokens(
        num_frequencies=16, include_input=True,
        d_model=d_model, nhead=8, layers=4
    ).to(device)

    loss, dx_pred, dx_gt = heun_step_loss(model, tokenizer, x1, x2, t, dt, cos_w=0.1)
    print("heun+tokens loss:", float(loss))
