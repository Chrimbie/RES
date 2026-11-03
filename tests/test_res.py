import tempfile
import unittest
from pathlib import Path
import numpy as np
from res_head import RESHead
from res_solution_extractor import extract

class TestRES(unittest.TestCase):
    def test_transfer_and_integrity(self):
        rng=np.random.default_rng(7)
        X=rng.normal(size=(140,5))
        y=(X[:,0]+0.5*X[:,1]>0).astype(int)
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/"source.res"
            b=Path(tmp)/"moved.res"
            extract(X[:100],y[:100],a,hidden=32,seed=9)
            original=RESHead(a)
            b.write_bytes(a.read_bytes())
            moved=RESHead(b)
            np.testing.assert_array_equal(original.predict(X[100:]),moved.predict(X[100:]))
            np.testing.assert_allclose(original.logits(X[100:]),moved.logits(X[100:]),rtol=0,atol=0)
            self.assertEqual(original.w0.shape,(5,32))
            with self.assertRaises(ValueError):
                moved.predict(np.ones((1,4)))

    def test_digits(self):
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        X,y=load_digits(return_X_y=True)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"digits.res"
            extract(Xtr,ytr,path,hidden=512,ridge=1,seed=42)
            head=RESHead(path)
            self.assertGreater(float(np.mean(head.predict(Xte)==yte)),0.90)

if __name__=="__main__":
    unittest.main()
