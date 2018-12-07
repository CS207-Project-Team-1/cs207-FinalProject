"""Test simple vectorized functions that use our Variable class with a numpy
backend"""
import ad
import pytest
import numpy as np

EPSILON = 0.0001

def equals(arr1, arr2):
    """Checks if arr1 and arr2 are within EPISILON distance of one another."""
    return np.power(arr1 - arr2, 2).sum() < EPSILON

""" Test simple multivar arithmetic equations for the Hessian """ 

def test_arithmetic_multivar_hessian():
    x, y, z = ad.Variable(), ad.Variable(), ad.Variable()
    f = 3 * x * x * y + (z - 1.0 / x)** 5