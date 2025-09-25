# src/eval_reconstruct.py
import os, argparse, yaml, torch, numpy as np
from datasets.mesh_dataset import TrianglePairs
from datasets.tri_pairs import TriPairsV2, TriPairsV3
from models.tri_predictor import TriFlow
from utils.metrics import l2_mean, chamfer
from utils.geometry import quantize_vertices
from utils.io import save_mesh


def load_flow(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['cfg']
    from models.tri_predictor import TriFlow
    model = TriFlow(
        cfg['model']['cond']['in_dim'], cfg['model']['cond']['hidden'], cfg['model']['cond']['depth'],
        cfg['model']['flow']['in_dim'], cfg['model']['flow']['hidden'], cfg['model']['flow']['depth'],
        act=cfg['model']['cond'].get('act','gelu')
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/defaults.yaml')
    ap.add_argument('--v2_ckpt', type=str, required=True)
    ap.add_argument('--v3_ckpt', type=str, required=True)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    device = cfg['project']['device']

    base = TrianglePairs(cfg['paths']['data_root'], split='test', backend=cfg['data']['backend'])

    v2_model, _ = load_flow(args.v2_ckpt, device)
    v3_model, _ = load_flow(args.v3_ckpt, device)

    # iterate triangles directly from base
    V_pred = []
    F_pred = []
    V_gt = []

    tri_id = 0
    l2_v2, l2_v3 = [], []

    for (v1, v2) in base.samples_v2:
        ((a,b), v3) = base.samples_v3[tri_id]
        tri_id = (tri_id + 1) % len(base.samples_v3)

        v1_t = torch.tensor(v1, dtype=torch.float32, device=device).unsqueeze(0)
        v2_t = torch.tensor(v2, dtype=torch.float32, device=device).unsqueeze(0)
        v3_t = torch.tensor(v3, dtype=torch.float32, device=device).unsqueeze(0)

        # predict v2 from v1
        cond2 = v1_t
        v2_hat = v2_model.sample(v1_t, cond2, steps=cfg['eval']['t_steps'])
        l2_v2.append(l2_mean(v2_hat, v2_t))

        # predict v3 from (v1, v2_hat)
        cond3 = torch.cat([v1_t, v2_hat], dim=-1)
        v3_hat = v3_model.sample(v2_hat, cond3, steps=cfg['eval']['t_steps'])
        l2_v3.append(l2_mean(v3_hat, v3_t))

        # collect for mesh export
        V_pred.extend([v1_t.squeeze(0).cpu().numpy(), v2_hat.squeeze(0).cpu().numpy(), v3_hat.squeeze(0).cpu().numpy()])
        F_pred.append([3*len(F_pred)+0, 3*len(F_pred)+1, 3*len(F_pred)+2])
        V_gt.extend([v1, v2, v3])

    Vp = np.asarray(V_pred, dtype=np.float64)
    Fp = np.asarray(F_pred, dtype=np.int64)
    Vp_u, inv = quantize_vertices(Vp, tol=cfg['data']['dedup_tol'])
    Fp_u = inv[Fp]

    os.makedirs(cfg['paths']['recon_dir'], exist_ok=True)
    save_mesh(os.path.join(cfg['paths']['recon_dir'], 'recon_pred.obj'), Vp_u, Fp_u)

    # simple chamfer vs. GT vertex set (unweighted)
    cham = chamfer(np.asarray(V_gt), Vp_u)

    print(f"L2(v2): {np.mean(l2_v2):.6f} | L2(v3): {np.mean(l2_v3):.6f} | Chamfer: {cham:.6f}")

if __name__ == '__main__':
    main()