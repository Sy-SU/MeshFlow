# src/visualize.py
import argparse, numpy as np
import open3d as o3d

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=str, required=True)
    args = ap.parse_args()
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])