"""Reproducible RES compatibility-helper experiment; numpy only."""
import hashlib, json
import numpy as np

rng=np.random.default_rng(17)
X=rng.normal(size=(9000,4)).astype(np.float32)
true_w=np.array([1.5,-2.0,.8,.3],np.float32)
y=(X@true_w>0)
train,cal,test=X[:4000],X[4000:5000],X[5000:]
ytrain,ycal,ytest=y[:4000],y[4000:5000],y[5000:]
w=np.zeros(4,np.float32)
for _ in range(700):
    p=1/(1+np.exp(-np.clip(train@w,-40,40)))
    w-=0.2*(train.T@(p-ytrain)/len(train))
raw=w.tobytes(); digest=hashlib.sha256(raw).hexdigest()
restored=np.frombuffer(raw,dtype=np.float32).copy()
source_features=['a','b','c','d']

def evaluate(name, input_features, transform, adapter=True, dtype=np.float32):
    original=np.array(restored,dtype=dtype)
    status='exact'
    if len(input_features)!=len(source_features) or set(input_features)!=set(source_features):
        return dict(case=name,status='incompatible',accuracy=None,reason='Missing/extra input features')
    if input_features!=source_features:
        status='mapped' if adapter else 'unmapped'
    def predict(x):
        ordered=x if not adapter else x[:,[input_features.index(k) for k in source_features]]
        return transform(ordered.astype(dtype))@original>0
    dest=test[:,[source_features.index(k) for k in input_features]]
    return dict(case=name,status=status,accuracy=round(float(np.mean(predict(dest)==ytest)*100),3),reason='')

identity=lambda x:x
nonlinear=lambda x:np.tanh(x)
permuted=['b','a','c','d']
results=[
    evaluate('Source linear',source_features,identity),
    evaluate('Nonlinear recipient, no helper',source_features,nonlinear),
    evaluate('Permuted nonlinear, no helper',permuted,nonlinear,False),
    evaluate('Permuted nonlinear, auto input mapper',permuted,nonlinear,True),
    evaluate('Permuted linear, auto input mapper',permuted,identity,True),
    evaluate('Nonlinear recipient, FP16 precision adapter',source_features,nonlinear,True,np.float16),
    evaluate('Missing feature: rejected',['a','b','c','x'],nonlinear),
]
A=np.array([[1,.2,0,0],[.5,1,0,0],[0,0,1,.4],[0,0,.2,1]],np.float32)
mixed=np.tanh(test@A)
mixed_accuracy=round(float(np.mean((mixed@restored>0)==ytest)*100),3)
recovered=np.arctanh(np.clip(mixed,-0.999999,0.999999))@np.linalg.inv(A)
recovered_accuracy=round(float(np.mean((recovered@restored>0)==ytest)*100),3)
results.extend([
    dict(case='Undocumented mixed inputs',status='unmapped',accuracy=mixed_accuracy,reason='Feature names alone cannot resolve mixing'),
    dict(case='Documented mixing + inverse adapter',status='mapped',accuracy=recovered_accuracy,reason='Requires known invertible preprocessing'),
])
assert raw==restored.tobytes() and hashlib.sha256(restored.tobytes()).hexdigest()==digest
assert results[4]['accuracy']==results[0]['accuracy']
assert results[3]['accuracy']==results[1]['accuracy']
assert results[6]['accuracy'] is None
print(json.dumps({'train_rows':len(train),'test_rows':len(test),'source_weight_sha256':digest,'weight_bytes_unchanged':True,'recipient_training_steps':0,'results':results},indent=2))
