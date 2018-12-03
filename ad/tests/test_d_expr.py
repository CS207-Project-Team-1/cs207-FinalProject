"""Tests for d_expr method"""
import ad
import pytest
import numpy as np
from math import pi


def test_high_order():
    x = ad.Variable("x")
    y = x ** 4
    yd5 = y.d_expr(5)
    yd9 = y.d_expr(9)
    assert np.isclose(yd5.eval({x: 4.0}), 0.0)
    assert np.isclose(yd5.eval({x: 10.0}), 0.0)
    assert np.isclose(yd9.eval({x: 4.0}), 0.0)
    assert np.isclose(yd9.eval({x: 10.0}), 0.0)


def test_power():
    x = ad.Variable("x")
    y = x ** 4
    yd1 = y.d_expr()
    yd2 = yd1.d_expr()
    yd3 = yd2.d_expr()
    yd4 = yd3.d_expr()
    assert np.isclose(yd1.eval({x: 2.0}), 32.0)
    assert np.isclose(yd2.eval({x: 2.0}), 48.0)
    assert np.isclose(yd3.eval({x: 2.0}), 48.0)
    assert np.isclose(yd4.eval({x: 2.0}), 24.0)
    assert np.isclose(yd4.eval({x: 3.0}), 24.0)


def test_sine():
    x = ad.Variable("x")
    y = ad.Sin(x)
    yd1 = y.d_expr()
    assert np.isclose(yd1.eval({x: pi / 2}), 0.0)
    assert np.isclose(yd1.eval({x:  0.0}), 1.0)
    assert np.isclose(yd1.d({x: 0.0}), 0.0)


def test_cosine():
    x = ad.Variable("x")
    y = ad.Cos(x)
    yd1 = y.d_expr()
    assert np.isclose(yd1.eval({x: pi / 2}), -1.0)
    assert np.isclose(yd1.eval({x:  0.0}), 0.0)
    assert np.isclose(yd1.d({x: 0.0}), -1.0)


def test_multiplication():
    x = ad.Variable("x")
    y = x * ad.Cos(x)
    yd1 = y.d_expr()
    assert np.isclose(yd1.eval({x: 0.0}), 1.0)


def test_addition():
    x = ad.Variable("x")
    y = x + ad.Sin(x)
    yd1 = y.d_expr()
    assert np.isclose(yd1.eval({x: 0.0}), 2.0)


def test_division():
    x = ad.Variable("x")
    y = 1 / x
    yd1 = y.d_expr()
    assert np.isclose(yd1.eval({x: 2.0}), -0.25)

    y = x / 2
    yd1 = y.d_expr()
    assert np.isclose(yd1.eval({x: 1.0}), 0.5)

    y = x / x
    yd1 = y.d_expr()
    assert np.isclose(yd1.eval({x: 100.0}), 0.0)


def test_substration_and_negation():
    x = ad.Variable("x")
    y = ad.Exp(2 * x) - x
    yd1 = y.d_expr()
    assert np.isclose(yd1.eval({x: 0.0}), 1.0)

    y = - ad.Exp(3 * x)
    yd1 = y.d_expr()
    assert np.isclose(yd1.eval({x: 0.0}), -3.0)