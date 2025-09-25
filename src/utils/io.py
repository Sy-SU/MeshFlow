# src/utils/io.py
import numpy as np, trimesh

def save_mesh(path, V, F):
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    mesh.export(path)