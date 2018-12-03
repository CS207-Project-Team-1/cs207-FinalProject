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

def test_variable_inheritance():
    x = ad.Variable()
    y = ad.Variable()

    f = x * y
    assert(x in f.dep_vars)
    assert(y in f.dep_vars)
    assert(len(f.dep_vars) == 2)
    assert(y not in x.dep_vars)
    assert(x not in y.dep_vars)


def test_variable_inheritance_three():
    x = ad.Variable()
    y = ad.Variable()
    z = ad.Variable()

    f = ad.Cos(x) * y
    g = ad.Sin(f) + z * z * ad.Log(z) + 1
    assert(x in f.dep_vars)
    assert(y in f.dep_vars)
    assert(len(f.dep_vars) == 2)
    assert(y not in x.dep_vars)
    assert(x not in y.dep_vars)
    assert(len(g.dep_vars) == 3)
    assert(x in g.dep_vars)
    assert(y in g.dep_vars)
    assert(z in g.dep_vars)
    assert(z not in f.dep_vars)
