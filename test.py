import numpy as np

z = np.load("outs/tri_dataset.npz", allow_pickle=True)

print("Keys:", list(z.keys()))

for split in ["train", "val", "test"]:
    for k in ["v1", "v2", "v3", "ctx", "mesh_id"]:
        key = f"{split}_{k}"
        print(key, z[key].shape, z[key].dtype)

print("Meta:", z["meta"].item())
