import pickle, torch, numpy as np
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": "cpu"})
with open("dataset_cartpole/picklefolder_test_indistr/batch_test_0_1.pkl", "rb") as f:
    xs, ys, cm, pm, pl = pickle.load(f)
torch.load = _orig
print("xs:", type(xs).__name__, getattr(xs, "shape", len(xs)))
print("ys:", type(ys).__name__, getattr(ys, "shape", len(ys)))
print("xs sample:", np.array(xs)[0][:3])
print("ys sample:", np.array(ys)[0][:3])
