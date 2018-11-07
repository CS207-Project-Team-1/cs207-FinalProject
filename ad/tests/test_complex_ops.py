"""Tests for complex, one dimensional binary and unary operations"""
import ad
import pytest
from math import pi
import numpy as np

def test_cosine_expression():
    a = ad.Variable('a')
    b = ad.Cos(a)
    assert np.isclose(b.eval({a: pi/2}), 0)

def test_sine_expression():
    a = ad.Variable('a')
    b = ad.Cos(a)
    assert b.eval({a: pi/2}) == 1

def test_tan_expression():
    a = ad.Variable('a')
    b = ad.Tan(a)
    assert b.eval({a: pi/4}) == 1

def test_sinh_expression():
    a = ad.Variable('a')
    b = ad.Sinh(a)
    assert b.eval({a: pi/4}) == 1

def test_cosh_expression():
    a = ad.Variable('a')
    b = ad.Cosh(a)
    assert b.eval({a: pi/4}) == 1

def test_tanh_expression():
    a = ad.Variable('a')
    b = ad.Tanh(a)
    assert b.eval({a: pi/4}) == 1