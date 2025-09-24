# MeshFlow
Flow matching model on mesh

```bash
conda env create -f environment.yml
conda activate meshflow
```

```bash
conda env update -f environment.yml --prune
```

```txt
MeshFlow/
├─ configs/                 # YAML：数据、模型、训练、评估
├─ src/
│  ├─ datasets/
│  │  ├─ mesh_dataset.py    # 读mesh→面片→排序→(v1,v2,v3)+局部上下文
│  │  └─ transforms.py      # 归一化、局部坐标、kNN、法线估计等
│  ├─ models/
│  │  ├─ flow_core.py       # CFM核心/向量场/采样器
│  │  ├─ cond_encoder.py    # 条件编码器(v1)/(v1,v2)+局部patch
│  │  └─ tri_predictor.py   # 两头：p(v2|v1), p(v3|v1,v2)
│  ├─ utils/
│  │  ├─ geometry.py        # 边/角/法线/重建/拓扑检查
│  │  ├─ knn.py             # kNN/球邻域检索
│  │  └─ io.py              # obj/ply/npz I/O
│  ├─ train_flow.py         # 训练脚本（多任务/双头）
│  ├─ eval_reconstruct.py   # 重建与指标评估
│  └─ visualize.py          # 可视化：三角形、法线、误差热力
├─ outs/
│  ├─ logs/                 # tensorboard/文本日志
│  ├─ ckpts/                # checkpoint：flow_v2, flow_v3
│  └─ reconstruct/          # 生成的mesh/诊断报告
├─ docs/
│  ├─ report_zh.md
│  └─ assets/
├─ scripts/
│  ├─ prepare_data.sh
│  ├─ train.sh
│  └─ eval.sh
├─ README.md
└─ requirements.txt
```