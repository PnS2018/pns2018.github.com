---
layout: default
---

Welcome to the first session of the course ['Deep Learning on Raspberry Pi'](../README.md).

In this session, we will revisit the basic concepts of linear algebra. Then, after a small introduction to the scripting language Python, we will familiarize ourselves with Numpy, a Python package used for scientific computing. Then we will familiarize ourselves with a few basics of symbolic computation. At the end of this session, there will be a few exercises which will further help understanding the concepts introduced.

## Linear Algebra

This section will only provide a brief introduction to linear algebra. For those of you unfamiliar with the concepts of linear algebra, it is strongly recommended that you spend some time with a text book or a complete course on linear algebra. A strongly recommended text book would be [Introduction to Linear Algbera](math.mit.edu/~gs/linearalgebra/) by Gilbert Strang.

Those of you who are well familiar with linear algebra may skip this section.

### Vector Space
A vector space $$V$$, over the set of real numbers $$\mathbb{R}$$, is a set equipped with two operations, addition `+` and multiplication `.`, subject to the conditions,
1. The set is closed under the addition operator, i.e. for any $$\vec{l}, \vec{m} \in V$$, $$\vec{l} + \vec{m} \in V$$.
2. The addition operation is commutative, i.e. for any $$\vec{l}, \vec{l} \in V$$, $$\vec{l} + \vec{m} = \vec{m} + \vec{l}$$.
3. The addition operation is associative, i.e. for any $$\vec{l}, \vec{m}, \vec{n} \in V$$, $$(\vec{l} + \vec{m}) + \vec{n} = \vec{l} + (\vec{m} + \vec{n})$$.
4. There exists a zero vector $$\vec{0} \in V$$, which is the identity element of addition, i.e. for any $$\vec{l} \in V$$, $$\vec{l} + \vec{0} = \vec{l}$$.
5. There should exist an inverse for every element in the set, i.e. for any $$\vec{l} \in V$$, there exists a $$\vec{m} \in V$$ such that $$\vec{l} + \vec{m} = \vec{0}$$.
6. The set is closed under the multiplication operation with any real valued scalar, i.e. for any $$\vec{l} \in V$$ and $$r\in \mathbb{R}$$, $$r.\vec{l} \in V$$.
7. The multiplication operation is distributive with the addition operator, i.e. for any $$gcr, s \in \mathbb{R}$$ and $$\vec{l}, \vec{m} \in V$$, $$(r + s).\vec{l} = r.\vec{l} + s.\vec{l}$$ and $$r.(\vec{l} + \vec{m}) = r.\vec{l} + r.\vec{m}$$.
8. The multiplication operation is compatible with the scalar multiplication operation, i.e. for any $$r, s \in \mathbb{R}$$ and any $$\vec{l} \in V$$, $$r.(s.\vec{l}) = (rs).\vec{l}$$.
9. There exists an identity element of scalar multiplication, i.e. for any $$\vec{l} \in V$$, $$1.\vec{l} = \vec{l}$$.


$$\mathbf{x}=\left[\begin{matrix}x_{1}\\ x_{2}\\ x_{3}\\ \vdots\\ x_{n}\end{matrix}\right]$$


```python
np.array([1, 2, 3, 4, 5, 6])  # a row vector that has 6 elements
```

### Matrix

$$\mathbf{A}=\left[\begin{matrix}A_{11} & A_{12} & \cdots & A_{1m} \\
A_{21} & A_{22} & \cdots & A_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
A_{n1} & A_{n2} & \cdots & A_{nm}\end{matrix}\right]$$

### Tensor
