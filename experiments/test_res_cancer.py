"""Reproduce RES cancer dataset benchmark using repository's extractor and head."""
import json
import tempfile
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from res_solution_extractor import extract
from res_head import RESHead

X,y=load_breast_cancer(return_X_y=True)
tr,te=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=42)
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/"cancer.res"
    extract(X[tr],y[tr],path,hidden=256,ridge=1.0,seed=42)
    head=RESHead(path)
    p=head.predict(X[te])
    moved=Path(tmp)/"transferred.res"
    moved.write_bytes(path.read_bytes())
    q=RESHead(moved).predict(X[te])
    baseline=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)).fit(X[tr],y[tr])
    report={"dataset":"Wisconsin Diagnostic Breast Cancer","test_samples":len(te),
            "accuracy":float(np.mean(p==y[te])),
            "logistic_regression_accuracy":float(np.mean(baseline.predict(X[te])==y[te])),
            "confusion_matrix":confusion_matrix(y[te],p,labels=[0,1]).tolist(),
            "transfer_exact":bool(np.array_equal(p,q)),
            "archive_bytes":path.stat().st_size}
    print(json.dumps(report,indent=2))
