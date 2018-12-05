"""Implementations of most simple trigonometic operations and other simple
unops that are used frequently"""
from .ad import Unop, Constant
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

    def _d_expr(self, var):
        if var not in self.dep_vars:
            return Constant(0)
        return Cos(self.expr1) * self.expr1._d_expr(var)

    def _d_n(self, n, feed_dict, e_cache_dict, d_cache_dict):
        if (id(self), n) in d_cache_dict:
            return d_cache_dict[(id(self), n)]
        if n == 0:
            res = self._eval(feed_dict, e_cache_dict)
            d_cache_dict[(id(self), 0)] = res
            return d_cache_dict[(id(self), 0)]
        res = 0
        cos_g = Cos(self.expr1)
        for i in range(1, n+1):
            if (id(self.expr1), i) not in d_cache_dict:
                g_i = self.expr1._d_n(i, feed_dict, e_cache_dict, d_cache_dict)
                d_cache_dict[(id(self.expr1), i)] = g_i
            if (id(cos_g), n-i) not in d_cache_dict:
                cos_g_ni = cos_g._d_n(n-i, feed_dict, e_cache_dict,
                                      d_cache_dict)
                d_cache_dict[(id(cos_g), n-i)] = cos_g_ni
            g_i = d_cache_dict[(id(self.expr1), i)]
            cos_g_ni = d_cache_dict[(id(cos_g), n-i)]
            res += (i * g_i * cos_g_ni)
        res /= n
        d_cache_dict[id(self), n] = res
        return d_cache_dict[id(self), n]


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

    def _d_expr(self, var):
        if var not in self.dep_vars:
            return Constant(0)
        return - Sin(self.expr1) * self.expr1._d_expr(var)

    def _d_n(self, n, feed_dict, e_cache_dict, d_cache_dict):
        if (id(self), n) in d_cache_dict:
            return d_cache_dict[(id(self), n)]
        if n == 0:
            res = self._eval(feed_dict, e_cache_dict)
            d_cache_dict[(id(self), 0)] = res
            return d_cache_dict[(id(self), 0)]
        res = 0
        sin_g = Sin(self.expr1)
        for i in range(1, n+1):
            if (id(self.expr1), i) not in d_cache_dict:
                g_i = self.expr1._d_n(i, feed_dict, e_cache_dict, d_cache_dict)
                d_cache_dict[(id(self.expr1), i)] = g_i
            if (id(sin_g), n-i) not in d_cache_dict:
                sin_g_ni = sin_g._d_n(n-i, feed_dict, e_cache_dict,
                                      d_cache_dict)
                d_cache_dict[(id(sin_g), n-i)] = sin_g_ni
            g_i = d_cache_dict[(id(self.expr1), i)]
            sin_g_ni = d_cache_dict[(id(sin_g), n-i)]
            res -= (i * g_i * sin_g_ni)
        res /= n
        d_cache_dict[id(self), n] = res
        return d_cache_dict[id(self), n]


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

    def _d_expr(self, var):
        if var not in self.dep_vars:
            return Constant(0)
        return 1.0 / (Cos(self.expr1) * Cos(self.expr1)) * \
               self.expr1._d_expr(var)


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

    def _d_expr(self, var):
        if var not in self.dep_vars:
            return Constant(var)
        return Cosh(self.expr1) * self.expr1._d_expr(var)


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

    def _d_expr(self, var):
        if var not in self.dep_vars:
            return Constant(0)
        return Sinh(self.expr1) * self.expr1._d_expr(var)


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

    def _d_expr(self, var):
        if var not in self.dep_vars:
            return Constant(0)
        return 1.0 / (Cosh(self.expr1) * Cosh(self.expr1)) * \
               self.expr1._d_expr(var)


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

    def _d_expr(self, var):
        if var not in self.dep_vars:
            return Constant(0)
        return self * self.expr1._d_expr(var)

    def _d_n(self, n, feed_dict, e_cache_dict, d_cache_dict):
        if (id(self), n) in d_cache_dict:
            return d_cache_dict[(id(self), n)]
        if n == 0:
            res = self._eval(feed_dict, e_cache_dict)
            d_cache_dict[(id(self), 0)] = res
            return d_cache_dict[(id(self), 0)]
        res = 0
        for i in range(1, n+1):
            if (id(self.expr1), i) not in d_cache_dict:
                g_i = self.expr1._d_n(i, feed_dict, e_cache_dict, d_cache_dict)
                d_cache_dict[(id(self.expr1), i)] = g_i
            if (id(self), n-i) not in d_cache_dict:
                exp_g_ni = self._d_n(n-i, feed_dict, e_cache_dict, d_cache_dict)
                d_cache_dict[(id(self), n-i)] = exp_g_ni
            g_i = d_cache_dict[(id(self.expr1), i)]
            exp_g_ni = d_cache_dict[(id(self), n-i)]
            res += (i * g_i * exp_g_ni)
        res /= n
        d_cache_dict[(id(self), n)] = res
        return d_cache_dict[(id(self), n)]


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

    def _d_expr(self, var):
        if var not in self.dep_vars:
            return Constant(0)
        return Constant(1.0) / self.expr1 * self.expr1._d_expr(var)

    def _d_n(self, n, feed_dict, e_cache_dict, d_cache_dict):
        if (id(self), n) in d_cache_dict:
            return d_cache_dict[(id(self), n)]
        if n == 0:
            res = self._eval(feed_dict, e_cache_dict)
            d_cache_dict[(id(self), 0)] = res
            return d_cache_dict[(id(self), 0)]
        res = 0
        for i in range(1, n):
            if (id(self), i) not in d_cache_dict:
                log_g_i = self._d_n(i, feed_dict, e_cache_dict, d_cache_dict)
                d_cache_dict[(id(self), i)] = log_g_i
            if (id(self.expr1), n-i) not in d_cache_dict:
                g_ni = self.expr1._d_n(n-i, feed_dict, e_cache_dict,
                                       d_cache_dict)
                d_cache_dict[(id(self.expr1), n-i)] = g_ni
            log_g_i = d_cache_dict[(id(self), i)]
            g_ni = d_cache_dict[(id(self.expr1), n-i)]
            res += (i * log_g_i * g_ni)
        res /= n
        if (id(self.expr1), n) not in d_cache_dict:
            g_n = self.expr1._d_n(n, feed_dict, e_cache_dict, d_cache_dict)
            d_cache_dict[(id(self.expr1), n)] = g_n
        g_n = d_cache_dict[(id(self.expr1), n)]
        res = g_n - res
        if (id(self.expr1), 0) not in d_cache_dict:
            g_0 = self.expr1._eval(feed_dict, e_cache_dict)
            d_cache_dict[(id(self.expr1), 0)] = g_0
        g_0 = d_cache_dict[(id(self.expr1), 0)]
        res /= g_0
        d_cache_dict[(id(self), n)] = res
        return d_cache_dict[(id(self), n)]