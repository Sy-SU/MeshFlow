 MeshFlow
Triangle-wise Conditional Flow Matching for Mesh Reconstruction.

## Setup
```bash
conda env create -f environment.yaml
conda activate meshflow
````

## 数据集处理
```bash
python src/datasets/prepare.py 
```

## 搜索超参数
```bash
chmod +x scripts/auto.sh
./scripts/auto.sh test
```
