"""Test simple vectorized functions that use our Variable class with a numpy
backend"""
import ad
import pytest
import numpy as np

EPSILON = 0.0001

def equals(arr1, arr2):
    """Checks if arr1 and arr2 are within EPISILON distance of one another."""
    return np.power(arr1 - arr2, 2).sum() < EPSILON

""" Test simple multivar arithmetic equations for the Hessian """ 

def test_arithmetic_multivar_hessian():
    x, y, z = ad.Variable(), ad.Variable(), ad.Variable()
    f = 3 * x * x * y + (z - 1.0 / x) ** 5
    h = f.hessian({x: 1, y: 2, z: 3})
    assert(equals(h[x][x], 12.0))
    assert(equals(h[y][y], 0))
    assert(equals(h[z][z], 160))
    assert(equals(h[x][y], 6))
    assert(equals(h[y][z], 0))
    assert(equals(h[x][z], 160))

def test_arithmetic_multivar_hessian_2():
    x, y, z = ad.Variable(), ad.Variable(), ad.Variable()
    f = -x * (x - y + 1.0) ** (-4) + (z / x) ** 2
    h = f.hessian({x: 1, y: 3, z: 5})
    assert(equals(h[x][x], 122))
    assert(equals(h[y][y], -20))
    assert(equals(h[z][z], 2))
    assert(equals(h[x][y], 24))
    assert(equals(h[y][z], 0))
    assert(equals(h[x][z], -20))

def test_trig_multivar_hessian_1():
    x, y, z = ad.Variable(), ad.Variable(), ad.Variable()
    f = ad.Sin(x * y) + z * ad.Cos(z * ad.Tan(1 / z)) 
    h = f.hessian({x: 1, y: 3, z: 5})
    assert(equals(h[x][x], -1.2700800725))
    assert(equals(h[y][y], -0.14112000805))
    assert(equals(h[z][z], -0.0050593837))
    assert(equals(h[x][y], -1.41335252))
    assert(equals(h[y][z], 0))
    assert(equals(h[x][z], 0))

def test_trig_multivar_hessian_2():
    x, y, z = ad.Variable(), ad.Variable(), ad.Variable()
    f = ad.Sinh(x * y) + (x + y) * (z ** 2) * ad.Cosh(z * ad.Tanh(1 / z)) 
    h = f.hessian({x: 1, y: 2, z: 3})
    assert(equals(h[x][x], 14.50744163138))
    assert(equals(h[y][y], 3.6268604078))
    assert(equals(h[z][z], 9.3022279093))
    assert(equals(h[x][y], 11.015916506))
    assert(equals(h[y][z], 9.2426242912))
    assert(equals(h[x][z], 9.2426242912))

def test_logexp_multivar_hessian_2():
    x, y, z = ad.Variable(), ad.Variable(), ad.Variable()
    f = ad.Sinh(ad.Exp(x - 3.0) * y) + ad.Log(y + x ** 2) * z * ad.Sin(x)
    h = f.hessian({x: 1, y: 3, z: 5})
    assert(equals(h[x][x], -1.5705707143))
    assert(equals(h[y][y], -0.2553174360))
    assert(equals(h[z][z], 0))
    assert(equals(h[x][y], 0.31902899408))
    assert(equals(h[y][z], 0.2103677462))
    assert(equals(h[x][z], 1.1697535323))
