"""
文件路径：src/datasets/dataloader.py
用途：该脚本包含了Mesh数据集的DataLoader，能够从保存的npz文件中加载顶点（vertices）和面片（faces）数据，并提供train/val/test数据集的批量数据。并且包含了对数据加载器的测试，确保其功能正确。如果点数小于等于200的Mesh会被丢弃，大于200的Mesh会随机采样200个点。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

class MeshDataset(Dataset):
    """从npz文件中加载Mesh数据的Dataset类"""
    
    def __init__(self, data_path, split='train', num_points=200):
        """
        Args:
            data_path (str): 数据文件路径，包含训练集、验证集或测试集
            split (str): 数据集划分，'train', 'val', 或 'test'
            num_points (int): 每个Mesh需要采样的顶点数量
        """
        self.data_path = data_path
        self.split = split
        self.num_points = num_points  # 每个Mesh需要采样的顶点数量
        self.file_paths = self._get_file_paths()
        
    def _get_file_paths(self):
        """根据数据集划分返回文件路径列表"""
        split_path = os.path.join(self.data_path, self.split)
        file_paths = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.npz')]
        return file_paths

    def __len__(self):
        """返回数据集的大小"""
        return len(self.file_paths)

    def _sample_points(self, vertices):
        """随机采样指定数量的顶点"""
        if vertices.shape[0] >= self.num_points:
            # 如果点数大于200，随机采样200个点
            sampled_indices = random.sample(range(vertices.shape[0]), self.num_points)
            sampled_vertices = vertices[sampled_indices]
        else:
            # 如果点数小于或等于200，保留所有点
            sampled_vertices = None
        
        return sampled_vertices

    def __getitem__(self, idx):
        """加载并返回数据样本"""
        file_path = self.file_paths[idx]
        data = np.load(file_path)

        # 获取每个文件的顶点数据和面片数据
        vertices = data['vertices']
        faces = data['faces']
        
        # 如果点数小于等于200，丢弃该文件
        if vertices.shape[0] <= self.num_points:
            return None  # 返回None，表示该文件被丢弃
        
        # 随机采样200个点
        vertices_sampled = self._sample_points(vertices)
        
        # 将数据转换为torch张量
        faces_tensor = torch.tensor(faces, dtype=torch.float32)
        vertices_tensor = torch.tensor(vertices_sampled, dtype=torch.float32)
        
        return vertices_tensor, faces_tensor

def collate_fn(batch):
    """处理batch中的None值（即丢弃的文件）"""
    # 过滤掉None值（即被丢弃的Mesh）
    batch = [item for item in batch if item is not None]
    vertices_batch, faces_batch = zip(*batch)
    
    # 将每个batch的样本堆叠成一个tensor
    vertices_batch = torch.stack(vertices_batch, dim=0)
    faces_batch = torch.stack(faces_batch, dim=0)
    
    return vertices_batch, faces_batch

def get_dataloader(data_path, batch_size=32, num_workers=0):
    """
    创建并返回训练、验证和测试的DataLoader
    """
    train_dataset = MeshDataset(data_path, split='train')
    val_dataset = MeshDataset(data_path, split='val')
    test_dataset = MeshDataset(data_path, split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader

def main():
    # 设置数据路径
    data_path = './outs/data'

    # 获取DataLoader
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(data_path, batch_size=2)

    # 打印数据格式
    print("Checking npz file structure...")

    # 打印一个npz文件的内容
    example_file = train_dataloader.dataset.file_paths[0]  # 获取第一个训练集文件路径
    data = np.load(example_file)

    print(f"\nLoaded {example_file}:")
    print("  Keys:", data.files)  # 打印npz文件中的所有键
    for key in data.files:
        print(f"  {key}: {data[key].shape}")  # 打印每个键对应的数据的形状

    # 运行一些简单的测试
    print("\nRunning simple tests on DataLoader...")

    # 测试train_dataloader是否能正确加载数据
    for vertices, faces in tqdm(train_dataloader, desc="Loading train data", unit="batch"):
        print(f"Batch loaded: vertices shape {vertices.shape}, faces shape {faces.shape}")
        break  # 只加载第一个批次进行测试

    print("\nDataLoader test completed.")

if __name__ == '__main__':
    main()
