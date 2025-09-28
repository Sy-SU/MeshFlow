"""
文件路径：src/datasets/dataloader.py
用途：面向面片（face）的 DataLoader。从 outs/data/ 下的 npz 文件读取 'faces'（形状为 [F, 3, 3]），
     对每个样本随机采样固定数量的面片 faces_per_mesh。若某样本面片数少于 faces_per_mesh，则丢弃。
     DataLoader 仅返回 faces，形状为 [B, faces_per_mesh, 3, 3]。在 main 中包含一次性加载测试与结构打印。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MeshFaceDataset(Dataset):
    """仅基于面片的 Mesh 数据集。固定采样 faces_per_mesh 个面片，不足则丢弃。"""

    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 faces_per_mesh: int = 1024,
                 shuffle_index: bool = True,
                 verbose: bool = False):
        """
        Args:
            data_path: 数据根目录（包含 train/val/test 文件夹）
            split:     'train' | 'val' | 'test'
            faces_per_mesh:  每个样本最终保留的面片数量（< 时丢弃；>= 时随机采样 faces_per_mesh）
            shuffle_index:   预筛选后是否打乱样本顺序
            verbose:         打印预筛选统计信息
        """
        self.data_path = data_path
        self.split = split
        self.faces_per_mesh = int(faces_per_mesh)

        split_path = os.path.join(self.data_path, self.split)
        all_files = [os.path.join(split_path, f)
                     for f in os.listdir(split_path) if f.endswith('.npz')]
        all_files.sort()

        kept, dropped = [], 0
        iterator = tqdm(all_files, desc=f"Indexing {split}", unit="file") if verbose else all_files

        for fp in iterator:
            try:
                with np.load(fp, allow_pickle=True) as arr:
                    if 'faces' not in arr:
                        dropped += 1
                        continue
                    faces = arr['faces']
                    # 兼容少数保存为 object 数组的情况
                    if isinstance(faces, np.ndarray) and faces.dtype == object:
                        try:
                            faces = np.stack(faces, axis=0)
                        except Exception:
                            dropped += 1
                            continue
                    if not isinstance(faces, np.ndarray):
                        dropped += 1
                        continue
                    if faces.ndim != 3 or faces.shape[1:] != (3, 3):
                        dropped += 1
                        continue
                    # 面片数量不足的样本直接丢弃
                    if faces.shape[0] < self.faces_per_mesh:
                        dropped += 1
                        continue
                    kept.append(fp)
            except Exception:
                dropped += 1
                continue

        if shuffle_index:
            rng = np.random.default_rng()
            rng.shuffle(kept)

        self.file_paths = kept
        self._stats = {"total": len(all_files), "kept": len(kept), "dropped": dropped}
        if verbose:
            print(f"[{split}] total={self._stats['total']} kept={self._stats['kept']} dropped={self._stats['dropped']}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        arr = np.load(fp, allow_pickle=True)
        faces = arr['faces']

        # 再次兜底处理 object 数组
        if isinstance(faces, np.ndarray) and faces.dtype == object:
            faces = np.stack(faces, axis=0)

        faces = faces.astype(np.float32)  # (F, 3, 3)
        F = faces.shape[0]

        # 随机采样固定数量的面片
        choice = np.random.choice(F, size=self.faces_per_mesh, replace=False)
        faces_sampled = faces[choice]  # (faces_per_mesh, 3, 3)

        faces_tensor = torch.from_numpy(faces_sampled)  # [faces_per_mesh, 3, 3]
        return faces_tensor


def collate_fn_faces(batch):
    """
    将 batch 中的 faces 堆叠：
      输入：batch 列表，元素为 [faces_per_mesh, 3, 3]
      输出：Tensor [B, faces_per_mesh, 3, 3]
    """
    return torch.stack(batch, dim=0)


def get_dataloader(data_path: str,
                   batch_size: int = 32,
                   num_workers: int = 0,
                   faces_per_mesh: int = 1024,
                   verbose: bool = False):
    """
    创建并返回训练、验证和测试的 DataLoader（仅返回 faces）。
    """
    train_dataset = MeshFaceDataset(
        data_path, split='train',
        faces_per_mesh=faces_per_mesh,
        shuffle_index=True, verbose=verbose
    )
    val_dataset = MeshFaceDataset(
        data_path, split='val',
        faces_per_mesh=faces_per_mesh,
        shuffle_index=False, verbose=verbose
    )
    test_dataset = MeshFaceDataset(
        data_path, split='test',
        faces_per_mesh=faces_per_mesh,
        shuffle_index=False, verbose=verbose
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn_faces, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn_faces, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn_faces, drop_last=False)
    return train_loader, val_loader, test_loader


def main():
    # 相对路径：outs/data/
    data_path = 'outs/data'
    faces_per_mesh = 1024
    batch_size = 123

    train_loader, val_loader, test_loader = get_dataloader(
        data_path, batch_size=batch_size, num_workers=0,
        faces_per_mesh=faces_per_mesh, verbose=True
    )

    if len(train_loader.dataset) == 0:
        print("No valid samples found in train split with current faces_per_mesh constraint.")
        return

    # 打印一个样本文件的键与原始形状（索引里的第一个文件）
    sample_fp = train_loader.dataset.file_paths[0]
    with np.load(sample_fp, allow_pickle=True) as d:
        print(f"\nLoaded {sample_fp}:")
        print("  Keys:", d.files)
        for k in d.files:
            print(f"  {k}: {d[k].shape}")

    # 运行一次 batch，确认 faces 维度对齐
    print("\nRunning a quick batch to verify aligned shapes...")
    for faces in tqdm(train_loader, desc="Loading train data", unit="batch"):
        print(f"Batch loaded: faces {tuple(faces.shape)}")
        # 预期：faces [B, {faces_per_mesh}, 3, 3]
        break

    print("\nDataLoader test completed.")

    # 取一个样本检查
    faces_item = MeshFaceDataset(data_path, 'train', faces_per_mesh).__getitem__(0)
    assert isinstance(faces_item, torch.Tensor)
    assert faces_item.dtype == torch.float32 and faces_item.shape == (faces_per_mesh, 3, 3)
    print("Single item OK:", tuple(faces_item.shape))

    # 取一个 batch 检查
    for faces_batch in train_loader:
        assert isinstance(faces_batch, torch.Tensor)
        assert faces_batch.ndim == 4 and faces_batch.shape[1:] == (faces_per_mesh, 3, 3)
        print("Batch OK:", tuple(faces_batch.shape))  # 预期: [B, faces_per_mesh, 3, 3]
        break



if __name__ == '__main__':
    main()
