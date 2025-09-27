import os
import numpy as np
import random
import shutil
from tqdm import tqdm

def read_obj(file_path):
    """从.obj文件中读取顶点坐标和面片信息"""
    vertices = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # 顶点行以 'v ' 开头
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])  # 获取x, y, z坐标
                vertices.append([x, y, z])
            elif line.startswith('f '):  # 面片行以 'f ' 开头
                parts = line.strip().split()
                # 读取每个面片的顶点索引
                face = [int(p.split('/')[0]) - 1 for p in parts[1:4]]  # 索引从 1 开始，减去 1
                faces.append(face)
    
    return np.array(vertices), np.array(faces)

def prepare_data(input_path, output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """将数据集划分为train/val/test并保存为npz格式"""
    # 创建目标目录
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)

    # 获取输入路径下所有的模型文件
    mesh_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.obj'):
                mesh_files.append(os.path.join(root, file))
    
    random.shuffle(mesh_files)  # 随机打乱文件顺序

    # 计算数据集划分的索引
    total_files = len(mesh_files)
    train_end = int(total_files * train_ratio)
    val_end = int(total_files * (train_ratio + val_ratio))

    # 划分数据集
    train_files = mesh_files[:train_end]
    val_files = mesh_files[train_end:val_end]
    test_files = mesh_files[val_end:]

    def save_files(file_list, split):
        """将文件保存到指定的目录，并保存为npz格式"""
        for file in tqdm(file_list, desc=f"Saving {split} files", unit="file"):
            # 获取爷爷目录名作为文件名
            grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(file)))

            # 读取模型顶点和面片数据
            vertices, faces = read_obj(file)

            # 存储每个面片的顶点索引和排序后的顶点
            face_v1_v2_v3 = []
            for face in faces:
                # 获取面片的顶点坐标
                v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                # 按照 z → y → x 排序顶点
                sorted_vertices = sorted([v1, v2, v3], key=lambda v: (v[2], v[1], v[0]))
                # 添加到面片列表
                sorted_indices = [face[0], face[1], face[2]]  # 对应顶点的索引
                face_v1_v2_v3.append((sorted_vertices[0], sorted_vertices[1], sorted_vertices[2]))

            # 保存为npz格式
            file_name = grandparent_dir + '.npz'  # 使用爷爷目录名作为文件名
            npz_path = os.path.join(output_path, split, file_name)
            # 每个npz文件保存v1, v2, v3和顶点坐标
            np.savez(npz_path, faces=face_v1_v2_v3, vertices=vertices)

    # 保存划分后的文件
    save_files(train_files, 'train')
    save_files(val_files, 'val')
    save_files(test_files, 'test')

    print(f"Data preparation done! Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

def main():
    input_path = '/root/autodl-fs/ShapeNetCore.v2.demo'
    output_path = 'outs/data'

    prepare_data(input_path, output_path)

if __name__ == '__main__':
    main()
