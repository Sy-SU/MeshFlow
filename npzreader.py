"""
文件路径：read_npz.py
用途：该脚本读取指定的 .npz 文件，并将文件中的数据结构（如 'faces' 和 'vertices'）完整地打印到 npz.txt 文件中。
"""

import os
import numpy as np

def save_npz_to_txt(file_path, output_file):
    """读取 .npz 文件并将完整内容保存到 npz.txt 文件"""
    try:
        # 加载 npz 文件
        data = np.load(file_path)
        with open(output_file, 'w') as f:
            f.write(f"Loaded {file_path}:\n")
            
            # 打印 npz 文件中的所有键
            f.write(f"  Keys: {data.files}\n")
            
            # 打印 faces 和 vertices 数据
            if 'faces' in data:
                f.write(f"  faces shape: {data['faces'].shape}\n")
                f.write(f"  faces data:\n{data['faces']}\n")
            else:
                f.write("  No 'faces' found in the file.\n")
            
            if 'vertices' in data:
                f.write(f"  vertices shape: {data['vertices'].shape}\n")
                f.write(f"  vertices data:\n{data['vertices']}\n")
            else:
                f.write("  No 'vertices' found in the file.\n")

        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"Error loading {file_path}: {e}")

def main():
    # 要读取的npz文件路径
    file_path = './outs/data/train/d8a1d270154b70e2aa1bf88d515bc6b2.npz'
    
    # 输出文件路径
    output_file = 'npz.txt'
    
    # 将npz文件内容保存到文本文件
    save_npz_to_txt(file_path, output_file)

if __name__ == '__main__':
    main()
