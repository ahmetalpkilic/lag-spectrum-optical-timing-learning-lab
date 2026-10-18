import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import eigvalsh
from diagnostic import lag_spectrum,features,scores
from modes import mode_sequence,lag_covariance,AFFINE
from threshold import rank_threshold
import lab
import repeat_model as r

class Tests(unittest.TestCase):
    def test_population_modes_and_affine_pencil(self):
        l=np.array([[2.,0],[.4,1.]])
        for modes in ([.4,.4],[-.3,.8]):
            a=lag_covariance(modes,l,1);b=l@l.T
            np.testing.assert_allclose(eigvalsh(a,b),sorted(modes),atol=1e-14)
            np.testing.assert_allclose(eigvalsh(AFFINE@a@AFFINE.T,AFFINE@b@AFFINE.T),sorted(modes),atol=1e-14)

    def test_sample_affine_and_shift_invariance(self):
        z=mode_sequence(np.random.default_rng(7).normal(size=(80,2)),[-.3,.8],np.eye(2),[0,0])
        np.testing.assert_allclose(lag_spectrum(z),lag_spectrum(z@AFFINE.T+[7,8]),atol=1e-13)
        self.assertTrue(np.all(abs(lag_spectrum(z))<=1))

    def test_independent_whitening_and_large_sample_recovery(self):
        z=mode_sequence(np.random.default_rng(90).normal(size=(30000,2)),[-.3,.8],np.array([[2.,0],[.5,1.]]),[4,7]);x=z-z.mean(0)
        chol=np.linalg.cholesky(x.T@x/len(x));w=np.linalg.solve(chol,x.T).T
        cross=w[:-1].T@w[1:]/len(w)
        np.testing.assert_allclose(lag_spectrum(z),np.linalg.eigvalsh((cross+cross.T)/2),atol=1e-13)
        np.testing.assert_allclose(lag_spectrum(z),[-.3,.8],atol=.025)

    def test_rank_bound_by_enumeration_and_ties(self):
        scores=np.arange(20)
        self.assertEqual(sum(scores[i]>rank_threshold(np.delete(scores,i),.2)[0] for i in range(20)),4)
        self.assertEqual(rank_threshold(np.ones(99)),(1.,95))
        self.assertEqual(rank_threshold(np.arange(999))[1],950)

    def test_budget_and_unpurchased_data(self):
        z=np.random.default_rng(12).normal(size=(40,10,2));old=np.array([0,.4]);fresh=np.array([.1,.7])
        for flag,m,refresh in [(False,10,0),(True,8,80)]:
            self.assertEqual(160+80+refresh+40*m,640)
            self.assertEqual(r.estimate(z[:,:m],(fresh if flag else old)[r.LABELS]).shape,(2,2,2))
            a=lab.policy_covariance(z,old,fresh,flag);changed=z.copy();changed[:,m:]=np.nan
            b=lab.policy_covariance(changed,old if not flag else [np.nan]*2,fresh if flag else [np.nan]*2,flag)
            np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(40,2,2))
            np.testing.assert_array_equal(a[::2],np.broadcast_to(a[0],(20,2,2)))

    def test_forced_branch_matches_controls(self):
        for threshold,control in [(100.,'never'),(-1.,'always')]:
            result=lab.evaluate(2,818,dict(scalar=threshold,spectrum=threshold))
            for row in result:
                for k in ('scalar','spectrum'):
                    self.assertEqual(row['errors_ms'][k],row['errors_ms'][control])

    def test_training_independence_and_determinism(self):
        a=lab.experiment(19,2,93);b=lab.experiment(19,3,93)
        self.assertEqual(a['training_scores'],b['training_scores']);self.assertEqual(a,lab.experiment(19,2,93))
        for row in a['rows']:
            self.assertTrue(all(np.isfinite(v['rmse_ms']) for v in row['methods'].values()))

    def test_invalid_data(self):
        for z in [np.zeros((20,2)),np.ones((5,2)),np.zeros((10,3)),np.full((10,2),np.nan)]:
            with self.assertRaises(ValueError):lag_spectrum(z)

if __name__=='__main__':unittest.main()
