#!/usr/bin/env bash
set -e
# Train v2
python -m src.train_flow --model configs/model_flow_v2.yaml --head v2
# Train v3 (after v2 converges or in parallel)
python -m src.train_flow --model configs/model_flow_v3.yaml --head v3