"""Tests for the basic, one dimensional binary and unary operations"""
import ad
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

def test_variable_negation():
    a = ad.Variable('a')
    b = ad.Negation(a)
    assert b.eval({a: 1}) == -1
