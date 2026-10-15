"""Verify optimized averaging against explicitly fitting every pair."""
import unittest
import numpy as np
from fit_pair_average import average_pair_maps


class PairAverageTests(unittest.TestCase):
    def test_matches_explicit_pair_fits(self):
        rng=np.random.default_rng(42)
        for N,n,d in [(20,5,4),(3,5,7),(6,4,5)]:
            U=rng.normal(size=(N,n,d))
            Y=rng.normal(size=(N,n,d))
            if N==6:
                U[:,:,-1]=U[:,:,0]*2
            actual,info=average_pair_maps(U,Y)
            explicit=[]
            for j in range(n):
                for i in range(j,n):
                    explicit.append(np.linalg.lstsq(U[:,j],Y[:,i],rcond=1e-5)[0].T)
            np.testing.assert_allclose(actual,np.mean(explicit,axis=0),rtol=1e-10,atol=1e-12)
            self.assertEqual(len(info),n)
            self.assertTrue(all(x['retained_rank']<=min(N,d) for x in info))

    def test_zero_and_one_position(self):
        A,info=average_pair_maps(np.zeros((3,2,4)),np.ones((3,2,4)))
        np.testing.assert_array_equal(A,np.zeros((4,4)))
        self.assertTrue(all(v['retained_rank']==0 for v in info))
        rng=np.random.default_rng(7)
        U=rng.normal(size=(20,1,4));Y=rng.normal(size=(20,1,4))
        A,_=average_pair_maps(U,Y)
        np.testing.assert_allclose(A,np.linalg.lstsq(U[:,0],Y[:,0],rcond=1e-5)[0].T,atol=1e-12)


if __name__=='__main__':
    unittest.main()
