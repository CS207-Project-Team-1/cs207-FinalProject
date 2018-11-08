"""Tests for complex, one dimensional binary and unary operations"""
import ad
import pytest
from math import pi
import numpy as np

"""
def test_variable_negation():
    a = ad.Variable('a')
    b = ad.Negation(a)
    assert b.eval({a:1}) == -1
"""

def test_sine_expression():
    a = ad.Variable('a')
    b = ad.Sin(a)
    assert np.isclose(b.eval({a: pi/2}), 1)
    assert np.isclose(b.d({a: pi/2}), 0)

def test_cosine_expression():
    a = ad.Variable('a')
    b = ad.Cos(a)
    assert np.isclose(b.eval({a: pi/2}), 0)
    assert np.isclose(b.d({a: pi/2}), -1)

def test_tan_expression():
    a = ad.Variable('a')
    b = ad.Tan(a)
    assert np.isclose(b.eval({a: pi/4}), 1)
    assert np.isclose(b.d({a: pi/4}), 2)

def test_sinh_expression():
    a = ad.Variable('a')
    b = ad.Sinh(a)
    assert np.isclose(b.eval({a: pi/4}), 0.8686709614860095)
    assert np.isclose(b.d({a: pi/4}), 1.3246090892520057)

def test_cosh_expression():
    a = ad.Variable('a')
    b = ad.Cosh(a)
    assert np.isclose(b.eval({a: pi/4}), 1.3246090892520057)
    assert np.isclose(b.d({a: pi/4}), 0.8686709614860095)

def test_tanh_expression():
    a = ad.Variable('a')
    b = ad.Tanh(a)
    assert np.isclose(b.eval({a: pi/4}), 0.6557942026326724)
    assert np.isclose(b.d({a: pi/4}), 0.5699339637933774)


def test_exp_expression():
    a = ad.Variable('a')
    b = ad.Exp(a)
    assert np.isclose(b.eval({a: 1}), 2.718281828459045)
    assert np.isclose(b.d({a: 1}), 2.718281828459045)

def test_variable_subtraction_derivative():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a - b
    d = b - a
    assert c.d({a:100, b:2}) == 0
    assert d.d({a:10, b:20}) == 0

def test_variable_multiplication_derivative():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a * b
    d = b * a
    assert c.d({a:100, b:2}) == 102
    assert d.d({a:10, b:20}) == 30

def test_variable_division():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a / b
    d = b / a
    assert c.d({a:100, b:2}) == -24.5
    assert d.d({a:10, b:20}) == -0.1