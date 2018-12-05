"""Test simple vectorized functions that use our Variable class with a numpy
backend"""
import ad
import pytest
import numpy as np

EPSILON = 0.0001

def equals(arr1, arr2):
    """Checks if arr1 and arr2 are within EPISILON distance of one another."""
    return np.power(arr1 - arr2, 2).sum() < EPSILON

def test_constant_hessian():
    """Checks the trivial case"""
    constant_5 = ad.Constant(5.1231)
    assert(constant_5.hessian({}) == 0)

def test_constant_simple_op_hessian():
    c1 = ad.Constant(50)
    c2 = ad.Constant(123)
    c3 = ad.Constant(500)
    f = c1 * c2 + c1 / c3 + (1 - c2) * (-c3)
    assert(f.hessian({}) == 0)

def test_single_variable_add_mult():
    x = ad.Variable()
    f = x * x * x * 5.0 + 5 * (-x) - x - x / 5.0
    assert(f.hessian({x: 0}) == 0) 
    assert(f.hessian({x: 1}) == 30.0) 

def test_single_variable_div():
    x = ad.Variable()
    f = 1.0 / (x + 1.0) + 5 - 2 * x 
    assert(equals(f.hessian({x: 1}), 0.25))
    assert(equals(f.hessian({x: 0}), 2.0))
