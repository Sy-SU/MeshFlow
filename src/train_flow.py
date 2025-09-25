# src/train_flow.py
import os, argparse, yaml, torch
from copy import deepcopy
from torch.utils.data import DataLoader
from datasets.mesh_dataset import TrianglePairs
from datasets.tri_pairs import TriPairsV2, TriPairsV3
from models.tri_predictor import TriFlow
from utils.train_utils import get_autocast, grad_clip_

def deep_update(base, extra):
    """Recursively update dict 'base' with 'extra'."""
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_configs(cli_config, model_config):
    """
    Load and deep-merge configs in this order:
      1) configs/defaults.yaml  (如果不存在，就跳过)
      2) configs/data.yaml
      3) configs/train.yaml
      4) cli_config              (e.g., configs/defaults.yaml 或你传的别的)
      5) model_config            (e.g., configs/model_flow_v2.yaml / v3)
    Later files override earlier ones (深度覆盖).
    """
    order = [
        "configs/defaults.yaml",
        "configs/data.yaml",
        "configs/train.yaml",
        cli_config,
        model_config,
    ]
    cfg = {}
    for p in order:
        if p and os.path.exists(p):
            with open(p, "r") as f:
                part = yaml.safe_load(f) or {}
            deep_update(cfg, part)
    return cfg

def make_model(cfg):
    cond_in = cfg['model']['cond']['in_dim']
    cond_hidden = cfg['model']['cond']['hidden']
    cond_depth = cfg['model']['cond']['depth']
    act = cfg['model']['cond'].get('act', 'gelu')

    flow_in = cfg['model']['flow']['in_dim']
    flow_hidden = cfg['model']['flow']['hidden']
    flow_depth = cfg['model']['flow']['depth']

    return TriFlow(cond_in, cond_hidden, cond_depth, flow_in, flow_hidden, flow_depth, act=act)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/defaults.yaml')
    ap.add_argument('--model', type=str, required=True, help='configs/model_flow_v2.yaml or model_flow_v3.yaml')
    ap.add_argument('--head', type=str, choices=['v2','v3'], required=True)
    args = ap.parse_args()

    cfg = load_configs(args.config, args.model)
    print("完成配置加载")

    torch.manual_seed(cfg['project']['seed'])
    device = cfg['project']['device']
    print("完成随机种子设置")

    base = TrianglePairs(
        cfg['paths']['data_root'],
        split='train',
        backend=cfg['data']['backend'],
        max_meshes=cfg['data']['max_meshes'],
        heads=('v2',) if args.head == 'v2' else ('v3',),
        debug=True,                # ✅ 打开可视化与日志
        debug_limit_tris=None,     # ✅ 先加载 5k 个三角形冒烟
        debug_log_every=1000       # 每 1000 个三角形打印一次内存/计数
    )
    if args.head == 'v2':
        ds = TriPairsV2(base)
    else:
        ds = TriPairsV3(base)
    dl = DataLoader(ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'], drop_last=True)

    print("完成数据集加载")

    model = make_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    print("完成模型加载")

    scaler = torch.cuda.amp.GradScaler(enabled=cfg['train']['amp'])

    global_step = 0
    for epoch in range(cfg['train']['epochs']):
        for it, batch in enumerate(dl):
            if args.head == 'v2':
                v1, v2 = [x.to(device) for x in batch]
                x0, x1 = v1, v2
                cond_in = v1
            else:
                v1, v2, v3 = [x.to(device) for x in batch]
                x0, x1 = v2, v3
                cond_in = torch.cat([v1, v2], dim=-1)

            t = torch.rand(x0.size(0), 1, device=device)

            with get_autocast(cfg['train']['amp']):
                loss = TriFlow.cfm_loss(model, x0, x1, cond_in, t)

            scaler.scale(loss).backward()
            grad_clip_(model.parameters(), cfg['train']['grad_clip'])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % cfg['logging']['print_every'] == 0:
                print(f"[e{epoch} i{it}] loss={loss.item():.6f}")

            if global_step % cfg['logging']['save_every'] == 0:
                os.makedirs(cfg['paths']['ckpt_dir'], exist_ok=True)
                head = args.head
                torch.save({'model': model.state_dict(), 'cfg': cfg}, os.path.join(cfg['paths']['ckpt_dir'], f'{cfg["model"]["name"]}_{head}_step{global_step}.pt'))

        # epoch end save
        os.makedirs(cfg['paths']['ckpt_dir'], exist_ok=True)
        torch.save({'model': model.state_dict(), 'cfg': cfg}, os.path.join(cfg['paths']['ckpt_dir'], f'{cfg["model"]["name"]}_{args.head}_e{epoch}.pt'))

if __name__ == '__main__':
    main()