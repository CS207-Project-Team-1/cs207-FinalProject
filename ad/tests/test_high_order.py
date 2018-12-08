"""Tests for d_n method"""
import ad
import pytest
import numpy as np
from math import pi

def test_unop():
    x = ad.Variable("x")

    y = -x
    assert np.isclose(y.d_n(n=0, val=2.0), -2.0)
    assert np.isclose(y.d_n(n=1, val=2.0), -1.0)
    assert np.isclose(y.d_n(n=2, val=2.0), 0.0)

    y = ad.Sin(2 * x)
    assert np.isclose(y.d_n(n=0, val=0.0), 0.0)
    assert np.isclose(y.d_n(n=1, val=0.0), 2.0)
    assert np.isclose(y.d_n(n=3, val=0.0), -8.0)

    y = ad.Exp(3 * x)
    assert np.isclose(y.d_n(n=0, val=0.0), 1.0)
    assert np.isclose(y.d_n(n=1, val=0.0), 3.0)
    assert np.isclose(y.d_n(n=3, val=0.0), 27.0)

    y = ad.Log(2 * x)
    assert np.isclose(y.d_n(n=0, val=0.5), 0.0)
    assert np.isclose(y.d_n(n=1, val=0.5), 2.0)
    assert np.isclose(y.d_n(n=3, val=0.5), 2.0/(0.5 ** 3))


def test_binop():
    x = ad.Variable("x")

    y = x ** 3
    assert np.isclose(y.d_n(n=0, val=2.0), 8.0)
    assert np.isclose(y.d_n(n=1, val=2.0), 12.0)
    assert np.isclose(y.d_n(n=3, val=2.0), 6.0)
    assert np.isclose(y.d_n(n=5, val=2.0), 0.0)

    y = x + x ** 3
    assert np.isclose(y.d_n(n=0, val=2.0), 10.0)
    assert np.isclose(y.d_n(n=1, val=2.0), 13.0)
    assert np.isclose(y.d_n(n=3, val=2.0), 6.0)
    assert np.isclose(y.d_n(n=5, val=2.0), 0.0)

    y = x - x ** 3
    assert np.isclose(y.d_n(n=0, val=2.0), -6.0)
    assert np.isclose(y.d_n(n=1, val=2.0), -11.0)
    assert np.isclose(y.d_n(n=3, val=2.0), -6.0)

    y = x * (x ** 2)
    assert np.isclose(y.d_n(n=0, val=2.0), 8.0)
    assert np.isclose(y.d_n(n=1, val=2.0), 12.0)
    assert np.isclose(y.d_n(n=3, val=2.0), 6.0)
    assert np.isclose(y.d_n(n=5, val=2.0), 0.0)

    y = (x ** 5) / (x ** 2)
    assert np.isclose(y.d_n(n=0, val=2.0), 8.0)
    assert np.isclose(y.d_n(n=1, val=2.0), 12.0)
    assert np.isclose(y.d_n(n=3, val=2.0), 6.0)
    assert np.isclose(y.d_n(n=5, val=2.0), 0.0)


def test_complex():
    x = ad.Variable("x")
    y = - 12 * ad.Cos(x ** 2) + 8 * (x ** 3) * ad.Sin(x ** 2)
    yd5 = y.d_expr(5)
    assert np.isclose(y.d_n(5, 2.0), yd5.eval({x: 2.0}))
