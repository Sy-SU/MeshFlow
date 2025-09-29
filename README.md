 MeshFlow
Triangle-wise Conditional Flow Matching for Mesh Reconstruction.

## Setup
```bash
conda env create -f environment.yml
conda activate meshflow
````

## 数据集处理
```bash
chmod +x scripts/prepare_data.sh
./scripts/prepare_data.sh
```

## 搜索超参数
```bash
chmod +x scripts/auto.sh
./scripts/auto.sh test
```
