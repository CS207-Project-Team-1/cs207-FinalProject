"""Tests the basic expression constructs used in the automatic differentiation
package."""
import ad
import pytest


def test_constant_eval():
    c = ad.Constant(5)
    assert c.eval({}) == 5
    assert c.d({}) == 0 

def test_variable_eval():
    x = ad.Variable('x')
    assert x.eval({x: 10.0}) == 10.0
    assert x.d({x: 10.0}) == 1.0

def test_invalid_variable_feed_raises():
    x = ad.Variable('x')
    with pytest.raises(ValueError):
        x.eval({'deadbeef': 5})

def test_power():
    x = ad.Variable('x')
    y = x ** 3 
    assert y.eval({x: 10.0}) == 1000.0
    assert y.d({x: 10.0}) == 300.0

def test_negation():
    x = ad.Variable('x')
    y = -x
    assert y.eval({x: 10.0}) == -10
    assert y.d({x: 10.0}) == -1