"""Tests for the basic, one dimensional binary and unary operations"""
import ad
import pytest

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