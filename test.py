#!/usr/bin/env python3
import os
import argparse
from collections import defaultdict

REQUIRED_ANY = ["model_normalized.obj", "model.obj"]  # 至少存在其一
OPTIONAL_ALL = ["model_normalized.mtl"]               # 建议有但非强制
OPTIONAL_ANY = ["model_normalized.solid.binvox", "model_normalized.surface.binvox"]

def is_nonempty_file(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False

def check_one_model(model_dir: str):
    """返回 (ok, reasons[])；ok=False 时 reasons 说明缺啥"""
    reasons = []
    models_dir = os.path.join(model_dir, "models")
    if not os.path.isdir(models_dir):
        reasons.append("missing_dir:models/")
        return False, reasons

    # 至少存在其一
    any_ok = False
    for fn in REQUIRED_ANY:
        fp = os.path.join(models_dir, fn)
        if is_nonempty_file(fp):
            any_ok = True
            break
    if not any_ok:
        reasons.append(f"missing_any:{'|'.join(REQUIRED_ANY)}")

    # 可选项：存在但大小为0也报告
    for fn in OPTIONAL_ALL:
        fp = os.path.join(models_dir, fn)
        if os.path.exists(fp) and not is_nonempty_file(fp):
            reasons.append(f"empty:{fn}")

    # 可选其一：两个 binvox 至少有一个存在则通过（不强制）
    opt_any_present = any(os.path.exists(os.path.join(models_dir, fn)) for fn in OPTIONAL_ANY)
    if not opt_any_present:
        # 不作为失败原因，只记录提示
        reasons.append(f"hint_no_binvox:{'|'.join(OPTIONAL_ANY)}")

    return len(reasons) == 0 or reasons == [f"hint_no_binvox:{'|'.join(OPTIONAL_ANY)}"], reasons

def scan_root(root: str, out_missing: str):
    by_synset = defaultdict(lambda: {"total":0, "ok":0, "missing":0})
    missing_list = []

    for synset in sorted(os.listdir(root)):
        synset_dir = os.path.join(root, synset)
        if not os.path.isdir(synset_dir):
            continue
        for model_id in sorted(os.listdir(synset_dir)):
            model_dir = os.path.join(synset_dir, model_id)
            if not os.path.isdir(model_dir):
                continue
            by_synset[synset]["total"] += 1
            ok, reasons = check_one_model(model_dir)
            if ok:
                by_synset[synset]["ok"] += 1
            else:
                by_synset[synset]["missing"] += 1
                missing_list.append((synset, model_id, ";".join(reasons)))

    # 打印汇总
    grand_total = sum(v["total"] for v in by_synset.values())
    grand_ok    = sum(v["ok"] for v in by_synset.values())
    grand_miss  = sum(v["missing"] for v in by_synset.values())

    print("==== Summary by synset ====")
    for s, v in sorted(by_synset.items()):
        print(f"{s}: total={v['total']}  ok={v['ok']}  missing={v['missing']}")
    print("==== Grand total ====")
    print(f"total={grand_total}  ok={grand_ok}  missing={grand_miss}")

    # 写缺失清单
    if missing_list:
        os.makedirs(os.path.dirname(out_missing), exist_ok=True)
        with open(out_missing, "w", encoding="utf-8") as f:
            for synset, mid, reason in missing_list:
                f.write(f"{synset}/{mid}  [{reason}]\n")
        print(f"Missing list saved to: {out_missing}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="ShapeNetCore.v2 根目录，如 /root/autodl-fs/ShapeNetCore.v2")
    ap.add_argument("--out", default="outs/missing_models.txt", help="缺失列表输出路径")
    args = ap.parse_args()
    scan_root(args.root, args.out)
