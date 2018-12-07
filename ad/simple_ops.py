"""Implementations of most simple trigonometic operations and other simple
unops that are used frequently"""
from .ad import Unop
import numpy as np

__all__ = ['Sin', 'Cos', 'Tan', 'Sinh', 'Cosh', 'Tanh', 'Exp', 'Log']


class Sin(Unop):
    """Trigonometric sine.

    Examples
    --------
    >>> import ad
    >>> x = ad.Variable('x')
    >>> y = ad.Sin(x)
    >>> y.eval({x: 1.0})
    0.8414709848078965
    >>> y.d({x: 1.0})
    0.54030230586813977
    """
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = np.sin(res1)
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = d1.get(var, 0) * np.cos(res1)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]

    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            d1 = self.expr1._d(feed_dict, e_cache, d_cache)
            v1 = self.expr1._eval(feed_dict, e_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = -np.sin(v1) * d1.get(var1, 0) * d1.get(var2, 0) \
                                      +np.cos(v1) * dxy1
            h_cache[id(self)] = ret
        return h_cache[id(self)]

class Cos(Unop):
    """Trigonometric cosine.

    Examples
    --------
    >>> import ad
    >>> x = ad.Variable('x')
    >>> y = ad.Cos(x)
    >>> y.eval({x: 1.0})
    0.54030230586813977
    >>> y.d({x: 1.0})
    -0.8414709848078965
    """
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = np.cos(res1)
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = - d1.get(var, 0) * np.sin(res1)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]

    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            d1 = self.expr1._d(feed_dict, e_cache, d_cache)
            v1 = self.expr1._eval(feed_dict, e_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = -np.cos(v1) * d1.get(var1, 0) * d1.get(var2, 0) \
                                      -np.sin(v1) * dxy1
            h_cache[id(self)] = ret
        return h_cache[id(self)]

class Tan(Unop):
    """Trigonometric tangent.

    Examples
    --------
    >>> import ad
    >>> x = ad.Variable('x')
    >>> y = ad.Tan(x)
    >>> y.eval({x: 1.0})
    1.5574077246549023
    >>> y.d({x: 1.0})
    3.42551882081476
    """
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = np.tan(res1)
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            tan_tmp = np.tan(res1)
            ret = {}
            for var in self.dep_vars:
                ret[var] = d1.get(var, 0) * (1 + tan_tmp * tan_tmp)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]
    
    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            d1 = self.expr1._d(feed_dict, e_cache, d_cache)
            v1 = self.expr1._eval(feed_dict, e_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = 2 * (np.sec(v1) ** 2) * np.tan(v1) * d1.get(var1, 0) * d1.get(var2, 0) \
                                      +(np.sec(v1) ** 2) * dxy1
            h_cache[id(self)] = ret
        return h_cache[id(self)]

class Sinh(Unop):
    """Hyperbolic sine.

    Examples
    --------
    >>> import ad
    >>> x = ad.Variable('x')
    >>> y = ad.Sinh(x)
    >>> y.eval({x: 1.0})
    1.1752011936438014
    >>> y.d({x: 1.0})
    1.5430806348152437
    """
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = np.sinh(res1)
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = d1.get(var, 0) * np.cosh(res1)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]

    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            d1 = self.expr1._d(feed_dict, e_cache, d_cache)
            v1 = self.expr1._eval(feed_dict, e_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = np.sinh(v1) * d1.get(var1, 0) * d1.get(var2, 0) + \
                                      np.cosh(v1) * dxy1
            h_cache[id(self)] = ret
        return h_cache[id(self)]

class Cosh(Unop):
    """Hyperbolic cosine.

    Examples
    --------
    >>> import ad
    >>> x = ad.Variable('x')
    >>> y = ad.Cosh(x)
    >>> y.eval({x: 1.0})
    1.5430806348152437
    >>> y.d({x: 1.0})
    1.1752011936438014
    """
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = np.cosh(res1)
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = d1.get(var, 0) * np.sinh(res1)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]

    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            d1 = self.expr1._d(feed_dict, e_cache, d_cache)
            v1 = self.expr1._eval(feed_dict, e_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = np.cosh(v1) * d1.get(var1, 0) * d1.get(var2, 0) + \
                                      np.sinh(v1) * dxy1
            h_cache[id(self)] = ret
        return h_cache[id(self)]


class Tanh(Unop):
    """Hyperbolic tangent.

    Examples
    --------
    >>> import ad
    >>> x = ad.Variable('x')
    >>> y = ad.Tanh(x)
    >>> y.eval({x: 1.0})
    0.76159415595576485
    >>> y.d({x: 1.0})
    0.41997434161402614
    """
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = np.tanh(res1)
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            tanh_tmp = np.tanh(res1)
            ret = {}
            for var in self.dep_vars:
                ret[var] = d1.get(var, 0) * (1 - tanh_tmp * tanh_tmp)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]

    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            d1 = self.expr1._d(feed_dict, e_cache, d_cache)
            v1 = self.expr1._eval(feed_dict, e_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = -2 * (np.sech(v1) ** 2) * np.tanh(v1) * d1.get(var1, 0) * d1.get(var2, 0) \
                                      +(np.sech(v1) ** 2) * dxy1
            h_cache[id(self)] = ret
        return h_cache[id(self)]
class Exp(Unop):
    """Exponential function in base e.

    Examples
    --------
    >>> import ad
    >>> x = ad.Variable('x')
    >>> y = ad.Exp(x)
    >>> y.eval({x: 1.0})
    2.7182818284590451
    >>> y.d({x: 1.0})
    2.7182818284590451
    """
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = np.exp(res1)
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = d1.get(var, 0) * np.exp(res1)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]

    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            d1 = self.expr1._d(feed_dict, e_cache, d_cache)
            v1 = self.expr1._eval(feed_dict, e_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = np.exp(v1) * d1.get(var1, 0) * d1.get(var2, 0) \
                                      +np.exp(v1) * dxy1
            h_cache[id(self)] = ret
        return h_cache[id(self)]

class Log(Unop):
    """Natural logarithm.
    The natural logarithm log is the inverse of the exponential function, so
    that log(exp(x)) = x. The natural logarithm is logarithm in base e.

    Examples
    --------
    >>> import ad
    >>> x = ad.Variable('x')
    >>> y = ad.Log(x)
    >>> y.eval({x: 1.0})
    0.0
    >>> y.d({x: 1.0})
    1.0
    """
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = np.log(res1)
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = d1.get(var, 0) / res1
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]

    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            d1 = self.expr1._d(feed_dict, e_cache, d_cache)
            v1 = self.expr1._eval(feed_dict, e_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = -(d1.get(var1, 0) * d1.get(var2, 0))/(v1 ** 2) \
                                      +dxy1 / v1
            h_cache[id(self)] = ret
        return h_cache[id(self)]
