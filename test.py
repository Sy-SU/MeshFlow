"""
文件路径：get_min_faces_count.py
用途：该脚本遍历所有npz文件，计算每个Mesh的面片数量，并将结果输出到result.txt文件中，按升序排序。如果面片数量小于100，则打印出该文件路径。
"""

import os
import numpy as np

def get_faces_count(data_path):
    """遍历所有 .npz 文件，获取每个文件的面片数量"""
    faces_count = []

    # 获取数据路径下所有的npz文件
    npz_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
    for npz_file in npz_files:
        npz_path = os.path.join(data_path, npz_file)
        data = np.load(npz_path)

        # 获取每个文件中的面片数量
        faces = data['faces']
        num_faces = faces.shape[0]  # 每个文件的面片数量

        faces_count.append((npz_file, num_faces))  # 保存文件名和对应的面片数量

        # 如果面片数量小于100，打印该文件路径
        if num_faces < 100:
            print(f"File with less than 100 faces: {npz_path}")

    # 按照面片数量升序排序
    faces_count.sort(key=lambda x: x[1])

    return faces_count

def save_faces_count(faces_count, output_file):
    """将每个Mesh的面片数量保存到result.txt文件"""
    with open(output_file, 'w') as f:
        for mesh, count in faces_count:
            f.write(f"{mesh}: {count}\n")
    print(f"Results saved to {output_file}")

def main():
    # 数据路径
    data_path = './outs/data/train'  # 请替换为你的数据路径
    output_file = 'result.txt'  # 结果保存的文件路径

    # 获取每个Mesh的面片数量
    faces_count = get_faces_count(data_path)

    # 保存到文件并升序排序
    save_faces_count(faces_count, output_file)

if __name__ == '__main__':
    main()
