"""Implementation of core structures used in our graph-based automatic
differentiation library. +, -, *, /, and exponentiation is also implemented
here to support simple operator overloading.
"""
import numpy as np

__all__ = ['Expression', 'Variable', 'Constant', 'Log']


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

    def d_expr(self, n=1):
        """Return n-th order derivative as an Expression.
        Scalar input only.
        """
        di = self
        for i in range(n):
            di = di._d_expr()
        return di

    def _d_expr(self):
        """Helper - Evaluates the derivative as an Expression."""
        raise NotImplementedError

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
        if name:
            self.name = str(name)
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

    def _d_expr(self):
        return Constant(1.0)

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

    def _d_expr(self):
        return Constant(0.0)


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

    def _d_expr(self):
        return - self.expr1._d_expr()


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

    def _d_expr(self):
        if isinstance(self.expr1, Constant):
            return np.log(self.expr1.val) * (self.expr1 ** self.expr2)
        elif isinstance(self.expr2, Constant):
            return self.expr2.val * (self.expr1 ** (self.expr2.val - 1))
        else:
            return self.expr2 * (self.expr1 ** (self.expr2 - 1)) * \
                   self.expr1._d_expr() + (self.expr1 ** self.expr2) * \
                   Log(self.exp1) * self.expr2._d_expr()


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

    def _d_expr(self):
        return self.expr1._d_expr() + self.expr2._d_expr()


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

    def _d_expr(self):
        return self.expr1._d_expr() - self.expr2._d_expr()


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

    def _d_expr(self):
        if isinstance(self.expr1, Constant):
            return self.expr1.val * self.expr2._d_expr()
        elif isinstance(self.expr2, Constant):
            return self.expr2.val * self.expr1._d_expr()
        else:
            return self.expr1 * self.expr2._d_expr() + self.expr2 * \
                   self.expr1._d_expr()


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

    def _d_expr(self):
        if isinstance(self.expr1, Constant):
            return - self.expr1.val * self.expr2._d_expr() / (self.expr2 *
                                                              self.expr2)
        elif isinstance(self.expr2, Constant):
            return self.expr1._d_expr() * (1.0 / self.expr2.val)
        else:
            return self.expr1._d_expr() / self.expr2 - self.expr1 * \
                   self.expr2._d_expr() / (self.expr2 * self.expr2)


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

    def _d_expr(self):
        return 1.0 / self.expr1 * self.expr1._d_expr()