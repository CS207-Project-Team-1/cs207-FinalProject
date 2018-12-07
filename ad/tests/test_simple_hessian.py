"""Test simple vectorized functions that use our Variable class with a numpy
backend"""
import ad
import pytest
import numpy as np

EPSILON = 0.0001

def equals(arr1, arr2):
    """Checks if arr1 and arr2 are within EPISILON distance of one another."""
    return np.power(arr1 - arr2, 2).sum() < EPSILON

""" Test simple 1D arithmetic equations for the Hessian """ 

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

"""Testing single variable trig functions"""

def test_single_variable_trig_sin():
    """analytic hessian is -25 sin (5 * x + 3)"""
    x = ad.Variable()
    f = ad.Sin(5 * x + 3)
    assert(equals(f.hessian({x: 1}), -25 * np.sin(5 * 1 + 3)))
    assert(equals(f.hessian({x: 2}), -25 * np.sin(5 * 2 + 3)))
    assert(equals(f.hessian({x: 3}), -25 * np.sin(5 * 3 + 3)))

def test_single_variable_trig_cos():
    """analytic hessian is -25 cos (5 * x + 3)"""
    x = ad.Variable()
    f = ad.Cos(5 * x + 3)
    assert(equals(f.hessian({x: 1}), -25 * np.cos(5 * 1 + 3)))
    assert(equals(f.hessian({x: 2}), -25 * np.cos(5 * 2 + 3)))
    assert(equals(f.hessian({x: 3}), -25 * np.cos(5 * 3 + 3)))
