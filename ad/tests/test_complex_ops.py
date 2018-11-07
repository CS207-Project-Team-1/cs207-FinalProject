"""Tests for complex, one dimensional binary and unary operations"""
import ad
import pytest
from math import pi
import numpy as np

def test_sine_expression():
    a = ad.Variable('a')
    b = ad.Sin(a)
    assert np.isclose(b.eval({a: pi/2}), 1)

def test_cosine_expression():
    a = ad.Variable('a')
    b = ad.Cos(a)
    assert np.isclose(b.eval({a: pi/2}), 0)

def test_tan_expression():
    a = ad.Variable('a')
    b = ad.Tan(a)
    assert np.isclose(b.eval({a: pi/4}), 1)

def test_sinh_expression():
    a = ad.Variable('a')
    b = ad.Sinh(a)
    assert np.isclose(b.eval({a: pi/4}), 0.8686709614860095)

def test_cosh_expression():
    a = ad.Variable('a')
    b = ad.Cosh(a)
    assert np.isclose(b.eval({a: pi/4}), 1.3246090892520057)

def test_tanh_expression():
    a = ad.Variable('a')
    b = ad.Tanh(a)
    assert np.isclose(b.eval({a: pi/4}), 0.6557942026326724)