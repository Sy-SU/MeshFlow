# src/train_flow.py
import os, argparse, yaml, torch
from torch.utils.data import DataLoader
from datasets.mesh_dataset import TrianglePairs
from datasets.tri_pairs import TriPairsV2, TriPairsV3
from models.tri_predictor import TriFlow
from utils.train_utils import get_autocast, grad_clip_


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

    cfg = {}
    for p in [args.config, 'configs/train.yaml', args.model]:
        with open(p, 'r') as f:
            cfg.update(yaml.safe_load(f))

    torch.manual_seed(cfg['project']['seed'])
    device = cfg['project']['device']

    base = TrianglePairs(cfg['paths']['data_root'], split='train', backend=cfg['data']['backend'], max_meshes=cfg['data']['max_meshes'])
    if args.head == 'v2':
        ds = TriPairsV2(base)
    else:
        ds = TriPairsV3(base)
    dl = DataLoader(ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'], drop_last=True)

    model = make_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

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