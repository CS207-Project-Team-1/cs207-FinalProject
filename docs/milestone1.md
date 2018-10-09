# Milestone 1

## Introduction

Differentiation is ubiquitous in almost all aspects of computer science, mathematics, and physics. It is used for numeric root-finding as in Newton's Method, and used for optimization with different forms of gradient descent.
However, calculating analytic derivatives is difficult and can lead to exponentially growing abstract syntax trees, which makes finding the derivative infeasible in many cases.
Similarly, calculating the derivative numerically using the limit definition runs into numeric instabilities due to limited machine precision.
Automatic differentiation addresses both of these issues - it uses the chain rule and the fact that computers calculate any function as a sequence of elementary operations to find the derivative.

## Background

## Usage

Since most expressions that require automatic differentiation are not constructed dynamically, we will start by building the computational graph statically. Then, to perform computations and get the derivative, we feed our computational graph some inputs. A sample usage would look like:

```python
>>> import ad
>>> x = ad.Variable('x')
>>> f = 10.0 + 5.0 * (x ** 2.0) - 3.0 * x
>>> f.eval({'x': 5.0})
120.0
>>> f.d
47.0
```

Multiple inputs could also be provided. For example,

```python
>>> x = ad.Variable('x')
>>> y = ad.Variable('y')
>>> f = x * y
>>> f.eval({'x': 5.0, 'y': 2.0})
10.0
>>> f.d
[2.0, 5.0]
```

## Software Organization

## Implementation
