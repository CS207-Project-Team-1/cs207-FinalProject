"""Tests for complex, one dimensional binary and unary operations"""
import ad
import pytest
from math import pi
import numpy as np

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


def test_log_expression():
    a = ad.Variable('a')
    b = ad.Log(a)
    assert np.isclose(b.eval({a: 5}), 1.6094379124341003)
    assert np.isclose(b.d({a: 5}), 0.2)

def test_variable_subtraction_derivative():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a - b
    d = b - a
    assert c.d({a:100, b:2}) == {a:1.0, b:-1.0}
    assert d.d({a:10, b:20}) == {a:-1.0, b:1.0}

def test_variable_addition_derivative():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a + b
    d = b + a
    assert c.d({a:100, b:2}) == {a:1.0, b:1.0}
    assert d.d({a:10, b:20}) == {a:1.0, b:1.0}

def test_variable_multiplication_derivative():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a * b
    d = b * a
    assert c.d({a:100, b:2}) == {a:2, b:100}
    assert d.d({a:10, b:20}) == {a:20, b:10}

def test_variable_division():
    a, b = ad.Variable('a'), ad.Variable('b')
    c = a / b
    d = b / a
    assert c.d({a:100, b:2}) == {a:0.5, b:-25.0}
    assert d.d({a:10, b:20}) == {a:-0.2, b:0.1}

def test_expression_exceptions():
    x = ad.Variable()
    c = ad.Expression()
    with pytest.raises(NotImplementedError):
        c.eval({})
    with pytest.raises(NotImplementedError):
        c.d({})
    with pytest.raises(NotImplementedError):
        c.hessian({})
    with pytest.raises(NotImplementedError):
        c._d_expr(x)
    with pytest.raises(NotImplementedError):
        c._d_n(1, {}, {}, {})

def test_variable_exceptions():
    x = ad.Variable('x')
    assert x.eval({'x':1}) == 1

def test_variable_exceptions_partial_derivative():
    x = ad.Variable('x')
    y = ad.Variable('y')
    assert x._d_expr(y).eval({x:1}) == 0

def test_variable_exceptions_repr():
    x = ad.Variable('x')
    y = ad.Variable()
    assert x.__repr__() == 'x'
    assert y.__repr__() == 'Var'

def test_constant_exceptions():
    x = ad.Constant('x')
    y = ad.Variable()
    assert x._d_expr(y).eval({}) == 0

def test_negation_exceptions():
    x = ad.Variable('x')
    y = -x
    f = y*2 + y
    assert f.d_n(1, 1) == -3

def test_hyperbolic_expressions():
    x = ad.Variable('x')
    y = ad.Variable('y')
    f1 = ad.Tan(x)
    f2 = ad.Sinh(x)
    f3 = ad.Cosh(x)
    f4 = ad.Tanh(x)
    f5 = ad.Exp(x)
    f6 = ad.Cos(x)
    assert f1._d_expr(y).eval({x:1}) == 0
    assert f1._d_expr(x).eval({x:1}) == f1.d({x: 1})
    assert f2._d_expr(y).eval({x:1}) == 0
    assert f2._d_expr(x).eval({x:1}) == f2.d({x: 1})
    assert f3._d_expr(y).eval({x:1}) == 0
    assert f3._d_expr(x).eval({x:1}) == f3.d({x: 1})
    assert f4._d_expr(y).eval({x:1}) == 0
    assert np.isclose(f4._d_expr(x).eval({x:1}), f4.d({x: 1}))
    assert f5._d_expr(y).eval({x:1}) == 0
    assert np.isclose(f5._d_expr(x).eval({x:1}), f5.d({x: 1}))
    assert f6._d_expr(y).eval({x:1}) == 0
    assert np.isclose(f6._d_expr(x).eval({x:1}), f6.d({x: 1}))

def test_exp_exceptions():
    x = ad.Variable('x')
    y = ad.Exp(x)
    z = ad.Cos(x)
    a = ad.Log(x)
    b = ad.Sin(x)
    f1 = y*2 + y
    f2 = z*2 + z
    f3 = a*2 + a
    f4 = b*2 + b
    assert np.isclose(f1.d_n(1, 1), 8.154845485377136)
    assert np.isclose(f2.d_n(1, 1), -2.5244129544236893)
    assert np.isclose(f3.d_n(1, 1), 3.0)
    assert np.isclose(f4.d_n(1, 1), 1.6209069176044193)

def test_inverse_trig_exceptions():
    x = ad.Variable('x')
    y = ad.Variable('y')
    f5 = ad.Arccos(x)
    f6 = ad.Arcsin(x)
    f7 = ad.Arctan(x)
    f8 = ad.Log(x)
    f9 = ad.Sin(x)
    assert f5._d_expr(y).eval({x:0}) == 0
    assert np.isclose(f5._d_expr(x).eval({x:0.5}), f5.d({x: 0.5}))
    assert f6._d_expr(y).eval({x:0}) == 0
    assert np.isclose(f6._d_expr(x).eval({x:0}), f6.d({x: 0}))
    assert f7._d_expr(y).eval({x:0}) == 0
    assert np.isclose(f7._d_expr(x).eval({x:0}), f7.d({x: 0}))
    assert f8._d_expr(y).eval({x:1}) == 0
    assert np.isclose(f8._d_expr(x).eval({x:2}), f8.d({x: 2}))
    assert f9._d_expr(y).eval({x:1}) == 0
    assert np.isclose(f9._d_expr(x).eval({x:2}), f9.d({x: 2}))