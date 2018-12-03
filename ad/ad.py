"""Implementation of core structures used in our graph-based automatic
differentiation library. +, -, *, /, and exponentiation is also implemented
here to support simple operator overloading.
"""
import numpy as np

__all__ = ['Expression', 'Variable', 'Constant']


class Expression(object):
    '''Base expression class that represents anything in our computational
    graph. Everything should be one of these.'''
    def __init__(self, grad=False):
        self.grad = grad
        self.children = []

    def eval(self, feed_dict):
        '''Evaluates the entire computation graph given a dictionary of
        variables mapped to values.'''
        return self._eval(feed_dict, dict())
    
    def _eval(self, feed_dict, cache_dict):
        '''Helper - Evaluates the computation graph recursively.'''
        raise NotImplementedError

    def d(self, feed_dict):
        '''Evaluates the derivative at the points given, returns to user'''
        res =  self._d(feed_dict, dict(), dict())
        if len(self.dep_vars) == 0:
            # No dependent variables - it is a constantb
            return 0
        if len(res) == 1:
            # This is the non-vectorized case, scalar func of scalar
            # Return a number, not a dictionary
            return list(res.values())[0]

        return res

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        '''Helper - Evaluates the differentiation products recursively.
        @param: feed_dict: dictionary mapping var names 
        @param: e_cache_dict: cache for previously evaluated values
        @param: d_cache_dict: cache for previously calculated derivatives
        '''
        raise NotImplementedError
    
    def hessian(self, feed_dict):
        '''Evaluates the hessian at the points given, returns to user as a 
        dictionary of dictionarys (to be indexed as [var1][var2] for the
        derivative with respect to var1 then var2)'''
        res = self._h(feed_dict, dict(), dict(), dict())
        return res
        # if len(self.dep_vars) == 0:
        #     return 0
        # elif len(self.dep_vars) == 1:
        #     # This is the 1D hessian case, so just a scalar
        #     return list(res.values())[0]

    def _h(self, feed_dict, e_cache_dict, d_cache_dict, h_cache_dict):
        '''Helper - Evaluates the differentiation products recursively.
        @param: feed_dict: dictionary mapping var names 
        @param: e_cache_dict: cache for previously evaluated values
        @param: d_cache_dict: cache for previously calculated derivatives
        @param: h_cache_dict: cache for previously calculated double derivatives 
        '''
        return NotImplementedError

    def __add__(self, other):
        try:
            # Propagate the need for gradient if one thing needs gradient
            # Need to call other.grad first since self.grad may shortcircuit
            return Addition(self, other, grad=(other.grad and self.grad))
        except AttributeError:
            return Addition(self, Constant(other), grad=self.grad)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            return Subtraction(self, other, grad=(other.grad and self.grad))
        except AttributeError:
            return Subtraction(self, Constant(other), grad=self.grad)

    def __rsub__(self, other):
        try:
            return Subtraction(other, self, grad=(other.grad and self.grad))
        except AttributeError:
            return Subtraction(Constant(other), self, grad=self.grad)

    def __mul__(self, other):
        try:
            return Multiplication(self, other, grad=(other.grad and self.grad))
        except AttributeError:
            return Multiplication(self, Constant(other), grad=self.grad)
    
    def __rmul__(self, other):
        # TODO: Multiplication not commutative if we enable matrix support
        return self.__mul__(other)
    
    def __truediv__(self, other):
        try:
            return Division(self, other, grad=(other.grad and self.grad))
        except AttributeError:
            return Division(self, Constant(other), grad=self.grad)

    def __rtruediv__(self, other):
        try:
            return Division(other, self, grad=(other.grad and self.grad))
        except AttributeError:
            return Division(Constant(other), self, grad=self.grad)

    def __neg__(self):
        return Negation(self, grad=self.grad)

    def __pow__(self, other):
        try:
            return Power(self, other, grad=(other.grad and self.grad))
        except AttributeError:
            return Power(self, Constant(other), grad=(self.grad))


class Variable(Expression):
    def __init__(self, name=None, grad=True):
        self.grad = grad
        self.name = None if not name else str(name)

        # A variable only depends on itself
        self.dep_vars = set([self])
    
    def _eval(self, feed_dict, cache_dict):
        # Check if the user specified either the object in feed_dict or
        # the name of the object in feed_dict
        if self in feed_dict:
            return feed_dict[self]
        elif self.name in feed_dict:
            return feed_dict[self.name]
        else:
            raise ValueError('Unbound variable %s' % self.name)
    
    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        return {self: 1.0}
    
    def _h(self, feed_dict, e_cache_dict, d_cache_dict, h_cache_dict):
        return {self: {self: 0}}

    def __repr__(self):
        if self.name:
            return self.name
        else:
            return "Var"


class Constant(Expression):
    '''Represents a constant.'''
    def __init__(self, val, grad=False):
        super().__init__(grad=grad)
        self.val = val
        self.dep_vars = set()
    
    def _eval(self, feed_dict, cache_dict):
        return self.val

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        return {}

    def _h(self, feed_dict, e_cache_dict, d_cache_dict, h_cache_dict):
        return {}


class Unop(Expression):
    """Utilities common to all unary operations in the form Op(a)

    Attributes
    ----------
    expr1: Expression
        Input of the unary function
    children: list of Expression
        The children of the unary function, i.e. expr1
    """
    def __init__(self, expr1, grad=False):
        """
        Parameters
        ----------
        expr1 : Expression
            Input of the unary function.
        grad : bool, optional
            If True, then allow the Expression to calculate the derivative.
        """
        super().__init__(grad=grad)
        self.expr1 = expr1
        self.children = [self.expr1]
        # Deep copy the set
        self.dep_vars = set(expr1.dep_vars)


class Negation(Unop):
    """Negation, in the form - A"""
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = -res1
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = -d1.get(var, 0)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]
    
    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the negations too
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    ret[var1][var2] = - h1.get(var1, {}).get(var2, 0)
            h_cache[id(self)] = ret
        return h_cache[id(self)]


class Binop(Expression):
    '''Utilities common to all binary operations in the form Op(a, b)'''
    def __init__(self, expr1, expr2, grad=False):
        super().__init__(grad=grad)
        try:
            expr1.grad
        except AttributeError:
            expr1 = Constant(expr1)
        try:
            expr2.grad
        except AttributeError:
            expr2 = Constant(expr1)
        self.expr1 = expr1
        self.expr2 = expr2
        self.children = [self.expr1, self.expr2]
        self.dep_vars = expr1.dep_vars | expr2.dep_vars


class Power(Binop):
    """Power function, the input is raised to the power of exponent.

    Examples
    --------
    >>> import ad
    >>> x = ad.Variable('x')
    >>> y = x ** 2
    >>> y.eval({x: 10.0})
    100.0
    >>> y.d({x: 10.0})
    20.0
    """
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            res2 = self.expr2._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = np.power(res1, res2)
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        """derivative is  y x^(y-1) x_dot + x^y log(x) y_dot"""
        if id(self) not in d_cache_dict:
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            res2 = self.expr2._eval(feed_dict, e_cache_dict)
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            d2 = self.expr2._d(feed_dict, e_cache_dict, d_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = res2 * np.power(res1, res2 - 1) * d1.get(var, 0) + \
                           np.power(res1, res2) * np.log(res1) * d2.get(var, 0)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]


class Addition(Binop):
    '''Addition, in the form A + B'''
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            res2 = self.expr2._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = res1 + res2
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            d2 = self.expr2._d(feed_dict, e_cache_dict, d_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = d1.get(var, 0) + d2.get(var, 0)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]
    
    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            h2 = self.expr2._h(feed_dict, e_cache, d_cache, h_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    dxy2 = h2.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = dxy1 + dxy2
            h_cache[id(self)] = ret
        return h_cache[id(self)]
            

class Subtraction(Binop):
    '''Subtraction, in the form A - B'''
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            res2 = self.expr2._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = res1 - res2
        return cache_dict[id(self)]
    
    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            d2 = self.expr2._d(feed_dict, e_cache_dict, d_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = d1.get(var, 0) - d2.get(var, 0)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]

    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            h2 = self.expr2._h(feed_dict, e_cache, d_cache, h_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    dxy2 = h2.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = dxy1 - dxy2
            h_cache[id(self)] = ret


class Multiplication(Binop):
    '''Multiplication, in the form A * B'''
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            res2 = self.expr2._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = res1 * res2
        return cache_dict[id(self)]
    
    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            d2 = self.expr2._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            res2 = self.expr2._eval(feed_dict, e_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = res1 * d2.get(var, 0) + res2 * d1.get(var, 0)
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]
    
    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            h2 = self.expr2._h(feed_dict, e_cache, d_cache, h_cache)
            d1 = self.expr1._d(feed_dict, e_cache, d_cache)
            d2 = self.expr2._d(feed_dict, e_cache, d_cache)
            v1 = self.expr1._eval(feed_dict, e_cache)
            v2 = self.expr2._eval(feed_dict, e_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    dxy1 = h1.get(var1, {}).get(var2, 0) 
                    dxy2 = h2.get(var1, {}).get(var2, 0) 
                    ret[var1][var2] = (d1.get(var1, 0) * d2.get(var2, 0) + 
                        d1.get(var2, 0) * d2.get(var1, 0) +
                        v1 * dxy2 + v2 * dxy1)
            h_cache[id(self)] = ret
        return h_cache[id(self)]


class Division(Binop):
    '''Division, in the form A / B'''
    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            res2 = self.expr2._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = res1 / res2
        return cache_dict[id(self)]
    
    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            d2 = self.expr2._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            res2 = self.expr2._eval(feed_dict, e_cache_dict)
            ret = {}
            for var in self.dep_vars:
                ret[var] = (d1.get(var, 0) / res2) - (d2.get(var, 0) * res1 /
                                                      (res2 * res2))
            d_cache_dict[id(self)] = ret
        return d_cache_dict[id(self)]

    def _h(self, feed_dict, e_cache, d_cache, h_cache):
        if id(self) not in h_cache:
            # Both dx^2 and dxdy are just the additions 
            h1 = self.expr1._h(feed_dict, e_cache, d_cache, h_cache)
            h2 = self.expr2._h(feed_dict, e_cache, d_cache, h_cache)
            d1 = self.expr1._d(feed_dict, e_cache, d_cache)
            d2 = self.expr2._d(feed_dict, e_cache, d_cache)
            f = self.expr1._eval(feed_dict, e_cache)
            g = self.expr2._eval(feed_dict, e_cache)
            ret = {var:{} for var in self.dep_vars}
            for var1 in self.dep_vars:
                for var2 in self.dep_vars:
                    fxy = h1.get(var1, {}).get(var2, 0) 
                    gxy = h2.get(var1, {}).get(var2, 0) 
                    fx, fy = d1.get(var1, 0), d1.get(var2, 0)
                    gx, gy = d2.get(var1, 0), d2.get(var2, 0)
                    term1 = - (gx * fy + gy * fx) / (g ** 2)
                    term2 = (2 * f * gy * gx) / (g ** 3)
                    term3 = (fxy / g) - (f * gxy) / (g ** 2)
                    ret[var1][var2] = term1 + term2 + term3
            h_cache[id(self)] = ret
        return h_cache[id(self)]
