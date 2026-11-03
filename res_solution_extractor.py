"""Dataset-to-RES extraction using random ReLU features and ridge regression."""
import argparse
import hashlib
import io
import json
import zipfile
import numpy as np
from res_head import RESHead

def extract(X,y,path,hidden=512,ridge=1.0,seed=42):
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y)
    if X.ndim!=2 or y.ndim!=1 or len(X)!=len(y) or not np.isfinite(X).all():
        raise ValueError("Invalid training data")
    if hidden<1 or ridge<=0:
        raise ValueError("Invalid hyperparameters")
    classes, encoded=np.unique(y,return_inverse=True)
    if len(classes)<2:
        raise ValueError("Need multiple classes")
    mean=X.mean(axis=0)
    scale=X.std(axis=0)
    scale[scale<1e-12]=1
    Z=(X-mean)/scale
    rng=np.random.default_rng(seed)
    w0=rng.normal(0,1/np.sqrt(X.shape[1]),(X.shape[1],hidden))
    b0=rng.normal(0,.5,hidden)
    H=np.maximum(Z@w0+b0,0)
    Y=np.eye(len(classes))[encoded]
    hc=H-H.mean(axis=0)
    yc=Y-Y.mean(axis=0)
    if hidden>len(X):
        w1=hc.T@np.linalg.solve(hc@hc.T+ridge*np.eye(len(X)),yc)
    else:
        w1=np.linalg.solve(hc.T@hc+ridge*np.eye(hidden),hc.T@yc)
    b1=Y.mean(axis=0)-H.mean(axis=0)@w1
    payload=io.BytesIO()
    np.savez(payload,w0=w0,b0=b0,w1=w1,b1=b1,mean=mean,scale=scale)
    raw=payload.getvalue()
    labels=[c.item() if isinstance(c,np.generic) else c for c in classes]
    if not all(isinstance(c,(str,int,float)) for c in labels):
        raise ValueError("Unsupported labels")
    manifest={"format":"res-head","version":"1.0","operator":"dense-relu-dense",
              "weights":"weights.npz","sha256":hashlib.sha256(raw).hexdigest(),
              "output_classes":labels,"input_features":X.shape[1],
              "hidden_features":hidden,"training_method":"random-ReLU-ridge",
              "training_examples":len(X),"seed":seed,"ridge":ridge}
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json",json.dumps(manifest,indent=2))
        archive.writestr("weights.npz",raw)
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--demo-digits",action="store_true")
    p.add_argument("--csv")
    p.add_argument("--out",default="solution.res")
    p.add_argument("--hidden",type=int,default=512)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--ridge",type=float,default=1.0)
    args=p.parse_args()
    if args.demo_digits:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        X,y=load_digits(return_X_y=True)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    elif args.csv:
        rows=np.genfromtxt(args.csv,delimiter=",",dtype=str,skip_header=1)
        rows=np.atleast_2d(rows)
        Xtr=rows[:,:-1].astype(float)
        ytr=rows[:,-1]
        Xte=yte=None
    else:
        p.error("Choose --demo-digits or --csv")
    extract(Xtr,ytr,args.out,hidden=args.hidden,ridge=args.ridge,seed=args.seed)
    head=RESHead(args.out)
    report={"archive":args.out,"training_accuracy":float(np.mean(head.predict(Xtr)==ytr))}
    if Xte is not None:
        report["held_out_accuracy"]=float(np.mean(head.predict(Xte)==yte))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
