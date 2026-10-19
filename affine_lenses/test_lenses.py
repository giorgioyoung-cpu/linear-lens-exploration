"""Numerical checks against explicit weighted least squares; requires only NumPy."""
import unittest
import numpy as np
from lens_core import stats,fit_stats,fit_paired,error

class AffineChecks(unittest.TestCase):
    def test_weighted_objectives(self):
        rng=np.random.default_rng(7)
        records=[(rng.normal(size=(n,3))+2,rng.normal(size=(n,3))-4) for n in (2,5,7)]
        for same in (False,True):
            xs=[];ys=[];weights=[]
            for u,y in records:
                pairs=[(i,j) for j in range(len(u)) for i in (range(j,len(u)) if not same else [j])]
                for i,j in pairs:
                    xs.append(u[j]);ys.append(y[i]);weights.append(1/len(records)/len(pairs))
            x,y,w=np.array(xs),np.array(ys),np.array(weights)
            design=np.column_stack((x,np.ones(len(x))))
            expected=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=None)[0]
            s=stats(records,same);A,b,_=fit_stats(s)
            np.testing.assert_allclose(A,expected[:-1].T,atol=1e-11)
            np.testing.assert_allclose(b,expected[-1],atol=1e-11)
            np.testing.assert_allclose(error(A,b,s),np.sum(w*np.sum((y-x@A.T-b)**2,axis=1)),atol=1e-11)
            # Also test arbitrary predictions, not only the minimizing map.
            B=rng.normal(size=(3,3));c=rng.normal(size=3)
            np.testing.assert_allclose(error(B,c,s),np.sum(w*np.sum((y-x@B.T-c)**2,axis=1)),atol=1e-10)

    def test_paired_explicit_and_translation(self):
        rng=np.random.default_rng(19)
        for N,d in ((9,3),(3,7)):
            u=rng.normal(size=(N,4,d))+3;y=rng.normal(size=(N,4,d))-2
            A,b,ranks=fit_paired(u,y)
            As=[];bs=[]
            for j in range(4):
                xc=u[:,j]-u[:,j].mean(axis=0)
                for i in range(j,4):
                    yc=y[:,i]-y[:,i].mean(axis=0)
                    B=(np.linalg.pinv(xc,rcond=1e-5)@yc).T
                    As.append(B);bs.append(y[:,i].mean(axis=0)-B@u[:,j].mean(axis=0))
            np.testing.assert_allclose(A,np.mean(As,axis=0),atol=1e-11)
            np.testing.assert_allclose(b,np.mean(bs,axis=0),atol=1e-11)
            shifted,_,_=fit_paired(u+11,y-9)
            np.testing.assert_allclose(A,shifted,atol=1e-11)
            self.assertTrue(all(r['rank']<=min(N-1,d) for r in ranks))

    def test_known_affine_and_degenerate(self):
        rng=np.random.default_rng(5);u=rng.normal(size=(50,3))+2
        B=rng.normal(size=(3,3));c=np.array([1.,2.,3.]);y=u@B.T+c
        A,b,_=fit_stats(stats([(u,y)],True))
        np.testing.assert_allclose(A,B,atol=1e-11);np.testing.assert_allclose(b,c,atol=1e-11)
        A,b,r=fit_stats(stats([(np.ones((5,3)),np.full((5,3),4.))],True))
        np.testing.assert_allclose(A,0);np.testing.assert_allclose(b,4);self.assertEqual(r['rank'],0)
        A,b,_=fit_paired(np.ones((4,3,2)),np.full((4,3,2),4.))
        np.testing.assert_allclose(A,0);np.testing.assert_allclose(b,4)

if __name__=='__main__': unittest.main()
