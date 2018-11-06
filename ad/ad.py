import numpy as np

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
        return self._d(feed_dict, dict(), dict())

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        '''Helper - Evaluates the differentiation products recursively.
        @param: feed_dict: dictionary mapping var names 
        @param: e_cache_dict: cache for previously evaluated values
        @param: d_cache_dict: cache for previously calculated derivatives
        '''
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


class Variable(Expression):
    def __init__(self, name, grad=True):
        self.grad = grad
        self.name = name
    
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
        return 1.0 


class Constant(Expression):
    '''Represents a constant.'''
    def __init__(self, val, grad=False):
        super().__init__(grad=grad)
        self.val = val
    
    def _eval(self, feed_dict, cache_dict):
        return self.val

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        return 0


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
            d_cache_dict[id(self)] = d1 * np.cos(res1)
        return d_cache_dict[id(self)]

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
            d_cache_dict[id(self)] = - d1 * np.sin(res1)
        return d_cache_dict[id(self)]

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
            d_cache_dict[id(self)] = d1 * (1 + tan_tmp * tan_tmp)
        return d_cache_dict[id(self)]

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
            d_cache_dict[id(self)] = d1 * np.cosh(res1)
        return d_cache_dict[id(self)]

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
            d_cache_dict[id(self)] = d1 * np.sinh(res1)
        return d_cache_dict[id(self)]

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
            d_cache_dict[id(self)] = d1 * (1 - tanh_tmp * tanh_tmp)
        return d_cache_dict[id(self)]

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
            d_cache_dict[id(self)] = d1 * np.exp(res1)
        return d_cache_dict[id(self)]

class Power(Unop):
    """Power function, the input is raised to the power of exponent.

    Examples
    --------
    >>> import ad
    >>> x = ad.Variable('x')
    >>> y = ad.Power(x, 2)
    >>> y.eval({x: 10.0})
    100.0
    >>> y.d({x: 10.0})
    20.0
    """
    def __init__(self, expr1, exponent, grad=False):
        super().__init__(expr1=expr1, grad=grad)
        self.exponent = exponent

    def _eval(self, feed_dict, cache_dict):
        if id(self) not in cache_dict:
            res1 = self.expr1._eval(feed_dict, cache_dict)
            cache_dict[id(self)] = np.power(res1, self.exponent)
        return cache_dict[id(self)]

    def _d(self, feed_dict, e_cache_dict, d_cache_dict):
        if id(self) not in d_cache_dict:
            d1 = self.expr1._d(feed_dict, e_cache_dict, d_cache_dict)
            res1 = self.expr1._eval(feed_dict, e_cache_dict)
            d_cache_dict[id(self)] = d1 * self.exponent \
                                     * np.power(res1, self.exponent-1)
        return d_cache_dict[id(self)]

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
            d_cache_dict[id(self)] = d1 / res1
        return d_cache_dict[id(self)]

class Binop(Expression):
    '''Utilities common to all binary operations in the form Op(a, b)'''
    def __init__(self, expr1, expr2, grad=False):
        super().__init__(grad=grad)
        self.expr1 = expr1
        self.expr2 = expr2
        self.children = [self.expr1, self.expr2]


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
            d_cache_dict[id(self)] = d1 + d2
        return d_cache_dict[id(self)]
            

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
            d_cache_dict[id(self)] = d1 - d2
        return d_cache_dict[id(self)]


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
            d_cache_dict[id(self)] = res1 * d2 + res2 * d1
        return d_cache_dict[id(self)]


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
            d_cache_dict[id(self)] = (d1 / res2) - (d2 * res1 / (res2 * res2))
        return d_cache_dict[id(self)]

