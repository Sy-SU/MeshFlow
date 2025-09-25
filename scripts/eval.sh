#!/usr/bin/env bash
set -e
python -m src.eval_reconstruct \
  --v2_ckpt outs/ckpts/flow_v2_v2_e19.pt \
  --v3_ckpt outs/ckpts/flow_v3_v3_e19.pt