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

def test_single_variable_trig_tan():
    """analytic hessian is -25 cos (5 * x + 3)"""
    x = ad.Variable()
    f = ad.Cos(5 * x + 3) * ad.Tan(x * x - 5)
    assert(equals(f.hessian({x: 1}), -48.05115800))
    assert(equals(f.hessian({x: 2}), -170.9403025))
    assert(equals(f.hessian({x: 3}), 218.2792716))

def test_single_variable_trig_hyperbolic():
    x = ad.Variable()
    f = x * x * ad.Cosh(0.01 * x + 0.1) + x * ad.Sinh(x * x - 4.0)
    assert(equals(f.hessian({x: 1}), 22.351093955))
    assert(equals(f.hessian({x: 2}), 14.02444322))
    assert(equals(f.hessian({x: 3}), 9351.7592912))

def test_single_variable_trig_hyperbolic_2():
    x = ad.Variable()
    # x^2  Cosh[Sin[x] + Tanh[Exp[3 * x] + Log[x]]]
    g = ad.Sin(x) + ad.Tanh(ad.Exp(3 * x) + ad.Log(x))
    f = x * x * ad.Cosh(g)
    assert(equals(f.hessian({x: 1}), 11.464317742))
    assert(equals(f.hessian({x: 2}), -13.704377252))

def test_single_variable_power_simple():
    x = ad.Variable()
    f = x ** 5 + x ** 3 - (x + 1.0 / x) ** 2
    assert(equals(f.hessian({x: 1}), 18.0))
    assert(equals(f.hessian({x: 2}), 169.625))
    assert(equals(f.hessian({x: 2.5}), 325.3464))

def test_double_variable_power_throws():
    x, y = ad.Variable(), ad.Variable()
    f = x ** (y + 5.0)
    g = x ** y
    with pytest.raises(NotImplementedError):
        g.hessian({x:1, y:1})
    with pytest.raises(NotImplementedError):
        f.hessian({x:1, y:1})
