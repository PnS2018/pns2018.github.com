---
layout: default
---

## Linear Regression

$$
\begin{align}
y=&w_{1}\cdot x_{1}+w_{2}\cdot x_{2}+\ldots+w_{i}\cdot x_{i}+\ldots+w_{n}+x_{n}+b \\
=&\sum_{i=1}^{n}w_{i}\cdot x_{i}+b
\end{align}
$$


```python
x = Input((10,), name="input_layer")
y = Linear(1, name="linear layer")
model = Model(x, y)
```

## Logistic Regression

```python
x = Input((10,), name="input_layer")
y = Linear(1, name="linear layer")
y = Activation("softmax")
model = Model(x, y)
```

## Stochastic Gradient Descent and its variants

$$\mathbf{w}^{\star}=\mathbf{w}-\alpha\frac{\partial\mathcal{L}}{\partial \mathbf{w}}$$
