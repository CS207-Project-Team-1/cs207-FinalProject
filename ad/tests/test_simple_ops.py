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

def test_constant_multiplication():
    a = ad.Constant(5)
    a2 = ad.Constant(-5)
    b = ad.Variable('b')
    c = a * b
    d = b * a2
    assert c.eval({b: 5}) == 25
    assert d.eval({b: 5}) == -25

def test_constant_multiplication():
    b = ad.Variable('b')
    c = 5 * b
    d = b * -5
    assert c.eval({b: 5}) == 25
    assert d.eval({b: 5}) == -25

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
