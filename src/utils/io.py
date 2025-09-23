import os
import trimesh

def load_mesh(path):
    """
    Return:
        V: (N, 3) float32
        F: (M, 3) int64  (triangle vertex indices)
    """
    mesh = trimesh.load(path, process=False)
    # 如果是场景，合并为单一 mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    assert mesh.faces.shape[1] == 3, "Only triangular meshes are supported."
    V = mesh.vertices.astype('float32')
    F = mesh.faces.astype('int64')
    return V, F

def iter_mesh_files(root, exts=('.obj', '.ply')):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                yield os.path.join(dirpath, fn)
