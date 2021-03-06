"""Tests for the basic, one dimensional binary and unary operations"""
import ad
import numpy as np
import pytest

def test_constant_addition():
    a = ad.Constant(5)
    b = ad.Variable('b')
    c = a + b
    assert c.eval({b: 5}) == 10

def test_constant_regular_addition():
    b = ad.Variable('b')
    c = b + 3
    d = 4 + b
    assert c.eval({b: 5}) == 8
    assert d.eval({b: 5}) == 9

def test_variable_addition():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a + b
    d = b + a
    assert c.eval({a: 100, b: 2}) == 102
    assert d.eval({a: 10, b: 20}) == 30

def test_constant_subtraction():
    a = ad.Constant(5)
    a2 = ad.Constant(-5)
    b = ad.Variable('b')
    c = a - b
    d = b - a2
    assert c.eval({b: 5}) == 0
    assert d.eval({b: 5}) == 10

def test_constant_regular_subtraction():
    b = ad.Variable('b')
    c = 5 - b
    d = b - 10
    assert c.eval({b: 5}) == 0
    assert d.eval({b: 5}) == -5

def test_variable_subtraction():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a - b
    d = b - a
    assert c.eval({a: 10, b: 2}) == 8
    assert d.eval({a: 10, b: 20}) == 10

def test_constant_multiplication():
    a = ad.Constant(5)
    a2 = ad.Constant(-5)
    b = ad.Variable('b')
    c = a * b
    d = b * a2
    assert c.eval({b: 5}) == 25
    assert d.eval({b: 5}) == -25

def test_constant_regular_multiplication():
    b = ad.Variable('b')
    c = 5 * b
    d = b * -5
    assert c.eval({b: 5}) == 25
    assert d.eval({b: 5}) == -25

def test_variable_multiplication():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a * b
    d = b * a
    assert c.eval({a: 100, b: 2}) == 200
    assert d.eval({a: 10, b: 20}) == 200

def test_constant_division():
    a = ad.Constant(5)
    a2 = ad.Constant(-5)
    b = ad.Variable('b')
    c = a / b
    d = b / a2
    assert c.eval({b: 5}) == 1
    assert d.eval({b: 5}) == -1

def test_constant_regular_division():
    b = ad.Variable('b')
    c = 5 / b
    d = b / -5
    assert c.eval({b: 5}) == 1
    assert d.eval({b: 5}) == -1

def test_variable_division():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a / b
    d = b / a
    assert c.eval({a: 100, b: 2}) == 50
    assert d.eval({a: 10, b: 20}) == 2

def test_inverse_trig():
    a, b = ad.Variable('a'), ad.Variable('b')

    c = ad.Arcsin(a / b)
    assert np.isclose(np.arcsin(0.5), c.eval({a: 1.0, b: 2.0}))
    d = ad.Arcsin(a)
    assert np.isclose(1.0/np.sqrt(1.0 - 0.5**2), d.d({a: 0.5}))

    c = ad.Arccos(a * b)
    assert np.isclose(np.arccos(0.25), c.eval({a: 0.5, b: 0.5}))
    d = ad.Arccos(a)
    assert np.isclose(-1.0 / np.sqrt(1.0 - 0.5 ** 2), d.d({a: 0.5}))

    c = ad.Arctan(a / b)
    assert np.isclose(np.arctan(0.5), c.eval({a: 1.0, b: 2.0}))
    d = ad.Arctan(a)
    assert np.isclose(1.0/(1 + 0.5**2), d.d({a: 0.5}))

def test_logistic():
    a = ad.Variable('a')
    f = ad.Logistic(a)
    assert f.eval({a:0}) == 0.5
    assert np.isclose(f.d({a:1}), 0.19661193324148188)

def test_sqrt():
    a = ad.Variable('a')
    f = ad.Sqrt(a)
    assert f.eval({a:1}) == 1.0
    assert f.d({a:1}) == 0.5

def test_logb():
    a = ad.Variable('a')
    f = ad.Logb(10, a)
    assert np.isclose(f.eval({a:2}), 0.30102999566398114)
    assert np.isclose(f.d({a:2}), 0.21714724095162588)
    
def test_power_base0():
    a = ad.Variable('a')
    c = a ** 2
    assert np.isclose(0, c.d_n(2, 0))
    assert np.isclose(0, c.d_n(4, 0))

    d = a ** (-1)
    with pytest.raises(ZeroDivisionError):
        d.d_n(2, 0)

    e = a ** (3.5)
    with pytest.raises(ZeroDivisionError):
        e.d_n(4, 0)

def test_power_constant_base():
    x = ad.Variable()
    f = 2 ** x
    assert np.isclose(f.eval({x: 5}), 32)
    assert np.isclose(f.eval({x: 5.2}), 2 ** 5.2)
