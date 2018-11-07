"""Test simple vectorized functions that use our Variable class with a numpy
backend"""
import ad
import pytest
import numpy as np

EPSILON = 0.0001

def equals(arr1, arr2):
    """Checks if arr1 and arr2 are within EPISILON distance of one another."""
    return np.power(arr1 - arr2, 2).sum() < EPSILON


def test_vector_constant_addition():
    a = ad.Constant(2.0 * np.ones(10))
    b = ad.Constant(3.0 * np.ones(10))
    a.eval({}) == 2.0 * np.ones(10)
    b.eval({}) == 3.0 * np.ones(10)
    c = a + b
    assert equals(c.eval({}), 5.0 * np.ones(10))

def test_vector_constant_subtraction():
    a = ad.Constant(2.0 * np.ones(10))
    b = ad.Constant(3.0 * np.ones(10))
    c = a - b
    assert equals(c.eval({}), - np.ones(10))
