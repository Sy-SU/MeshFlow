 MeshFlow
Triangle-wise Conditional Flow Matching for Mesh Reconstruction.

## Setup
```bash
conda env create -f environment.yml
conda activate meshflow
````

## Train

```bash
bash scripts/train.sh
```

## Evaluate

```bash
bash scripts/eval.sh
```

## Data

* Place ShapeNetCore.v2 under `paths.data_root`.
* Or provide an `npz` tri dataset with keys `{train,val,test}_{v1,v2,v3}`.

```txt
MeshFlow/
├─ configs/
│ ├─ defaults.yaml
│ ├─ data.yaml
│ ├─ model_flow_v2.yaml
│ ├─ model_flow_v3.yaml
│ ├─ train.yaml
│ └─ eval.yaml
├─ src/
│ ├─ datasets/
│ │ ├─ mesh_dataset.py
│ │ ├─ tri_pairs.py
│ │ └─ transforms.py
│ ├─ models/
│ │ ├─ flow_core.py
│ │ ├─ cond_encoder.py
│ │ └─ tri_predictor.py
│ ├─ utils/
│ │ ├─ geometry.py
│ │ ├─ io.py
│ │ ├─ knn.py
│ │ ├─ metrics.py
│ │ ├─ seed.py
│ │ └─ train_utils.py
│ ├─ train_flow.py
│ ├─ eval_reconstruct.py
│ └─ visualize.py
├─ outs/
│ ├─ logs/
│ ├─ ckpts/
│ └─ reconstruct/
├─ docs/
│ ├─ report_zh.md
│ └─ assets/
├─ scripts/
│ ├─ prepare_data.sh
│ ├─ train.sh
│ └─ eval.sh
├─ README.md
└─ environment.yml
```