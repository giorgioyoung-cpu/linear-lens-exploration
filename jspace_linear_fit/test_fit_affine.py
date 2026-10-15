"""Numerical checks; run: python test_fit_affine.py"""
import unittest
import numpy as np
from fit_affine import sequence_moments, fit_linear, prediction_error


class LinearFitChecks(unittest.TestCase):
    def test_matches_explicit_weighted_least_squares(self):
        rng = np.random.default_rng(42)
        passages = [(rng.normal(size=(n, 4)), rng.normal(size=(n, 4)))
                    for n in (3, 7, 11)]
        moments = [sequence_moments(U, Y) for U, Y in passages]
        S, C, q = [sum(m[k] for m in moments) / len(moments) for k in range(3)]
        # Independent reference: enumerate every pair and solve via SVD least squares.
        Xrows, Yrows = [], []
        for U, Y in passages:
            n = len(U)
            weight = 1 / (len(passages) * n * (n + 1) / 2)
            for i in range(n):
                for j in range(i + 1):
                    Xrows.append(np.sqrt(weight) * U[j])
                    Yrows.append(np.sqrt(weight) * Y[i])
        X, Y = np.array(Xrows), np.array(Yrows)
        np.testing.assert_allclose(S, X.T @ X, atol=1e-12)
        np.testing.assert_allclose(C, Y.T @ X, atol=1e-12)
        self.assertAlmostEqual(q, np.sum(Y * Y))
        A, _, _ = fit_linear(S, C)
        reference = np.linalg.lstsq(X, Y, rcond=None)[0].T
        np.testing.assert_allclose(A, reference, rtol=1e-10, atol=1e-12)
        for B in (A, np.eye(4), np.zeros((4, 4)), rng.normal(size=(4, 4))):
            self.assertAlmostEqual(prediction_error(B, S, C, q), np.sum((Y-X@B.T)**2))

    def test_singular_minimum_norm_and_no_intercept(self):
        U = np.array([[1., 2., 0.], [2., 4., 0.], [3., 6., 0.]])
        Y = np.array([[3., 1., 2.], [4., 0., 3.], [7., 2., 1.]])
        S, C, q = sequence_moments(U, Y)
        X = np.array([U[j] for i in range(3) for j in range(i+1)])
        targets = np.array([Y[i] for i in range(3) for j in range(i+1)])
        A, _, info = fit_linear(S, C)
        self.assertEqual(info['retained_rank'], 1)
        np.testing.assert_allclose(A, np.linalg.lstsq(X, targets, rcond=1e-10)[0].T, atol=1e-12)
        np.testing.assert_allclose(A @ np.zeros(3), np.zeros(3))
        self.assertAlmostEqual(prediction_error(A, S, C, q), np.mean(np.sum((targets-X@A.T)**2,axis=1)))

    def test_zero_inputs(self):
        S, C, q = sequence_moments(np.zeros((4, 3)), np.ones((4, 3)))
        A, _, info = fit_linear(S, C)
        self.assertEqual(info['retained_rank'], 0)
        np.testing.assert_array_equal(A, np.zeros((3,3)))
        self.assertEqual(prediction_error(A, S, C, q), 3.)


if __name__ == '__main__':
    unittest.main()
