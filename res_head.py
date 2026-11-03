"""RES direct execution head: NumPy runtime, no training-model object."""
import io
import json
import hashlib
import zipfile
import numpy as np

class RESHead:
    def __init__(self, path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or len(names) > 8:
                raise ValueError("Invalid archive")
            if "manifest.json" not in names or archive.getinfo("manifest.json").file_size > 65536:
                raise ValueError("Invalid manifest")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != "res-head" or manifest.get("version") != "1.0":
                raise ValueError("Unsupported RES version")
            if manifest.get("operator") != "dense-relu-dense":
                raise ValueError("Unsupported execution operator")
            member = manifest.get("weights")
            if member != "weights.npz" or member not in names:
                raise ValueError("Invalid weights member")
            if archive.getinfo(member).file_size > 128 * 1024 * 1024:
                raise ValueError("Weights too large")
            raw = archive.read(member)
            if hashlib.sha256(raw).hexdigest() != manifest.get("sha256"):
                raise ValueError("Weights integrity check failed")
        with np.load(io.BytesIO(raw), allow_pickle=False) as tensors:
            required = ("w0", "b0", "w1", "b1", "mean", "scale")
            if not all(k in tensors.files for k in required):
                raise ValueError("Missing tensors")
            a = {k: tensors[k].copy() for k in required}
        w0,b0,w1,b1 = (a[k] for k in ("w0","b0","w1","b1"))
        mean,scale = a["mean"],a["scale"]
        if any(t.dtype.kind != "f" or not np.isfinite(t).all() for t in a.values()):
            raise ValueError("Non-finite or non-floating tensors")
        if w0.ndim != 2 or w1.ndim != 2 or b0.shape != (w0.shape[1],) or w1.shape[0] != w0.shape[1] or b1.shape != (w1.shape[1],):
            raise ValueError("Incompatible weight shapes")
        if mean.shape != (w0.shape[0],) or scale.shape != mean.shape or np.any(scale == 0):
            raise ValueError("Invalid normalization")
        classes = manifest.get("output_classes")
        if not isinstance(classes, list) or len(classes) != len(b1):
            raise ValueError("Invalid output classes")
        self.w0,self.b0,self.w1,self.b1 = w0,b0,w1,b1
        self.mean,self.scale,self.classes = mean,scale,np.asarray(classes)
        self.manifest = manifest

    def logits(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None,:]
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Incompatible input")
        x = (x-self.mean)/self.scale
        return np.maximum(x@self.w0+self.b0,0)@self.w1+self.b1

    def predict(self, x):
        return self.classes[np.argmax(self.logits(x),axis=1)]

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("solution")
    p.add_argument("--input",help="NumPy .npy feature matrix")
    args=p.parse_args()
    head=RESHead(args.solution)
    if args.input:
        print(head.predict(np.load(args.input,allow_pickle=False)).tolist())
    else:
        print(json.dumps(head.manifest,indent=2))
