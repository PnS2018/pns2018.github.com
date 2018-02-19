---
layout: default
---

## Linear Regression

$$
\begin{aligned}
y=&w_{1}\cdot x_{1}+w_{2}\cdot x_{2}+\ldots+w_{i}\cdot x_{i}+\ldots+w_{n}+x_{n}+b \\
=&\sum_{i=1}^{n}w_{i}\cdot x_{i}+b
\end{aligned}
$$


```python
x = Input((10,), name="input_layer")
y = Linear(1, name="linear layer")
model = Model(x, y)
```

## Logistic Regression


$$
\begin{aligned}
f(\mathbf{x}) =& \frac{1}{1-\exp(-\mathbf{W}^{\top}\mathbf{x}} \\
\Pr(y=1|\mathbf{x}) =& f(\mathbf{x}) \\
\Pr(y=0|\mathbf{x}) =& 1-f(\mathbf{x})
\end{aligned}
$$

$$
\mathcal{L}(\mathbf{X}, \mathbf{y}|\mathbf{W}) = -\frac{1}{N}\sum_{i}\left(y^{(i)} \log(\Pr(y=1|\mathbf{x}^{(i)}))+(1-y^{(i)})\log(\Pr(y=0|\mathbf{x}^{(i)}))\right)
$$

$$
\mathbf{W}^{\star}=\arg\min_{\mathbf{W}}\mathcal{L}(\mathbf{X}, \mathbf{y})
$$

```python
x = Input((10,), name="input_layer")
y = Linear(1, name="linear layer")
y = Activation("softmax")
model = Model(x, y)
```

## Stochastic Gradient Descent and its variants

$$\mathbf{w}^{\star}=\mathbf{w}-\alpha\frac{\partial\mathcal{L}}{\partial \mathbf{w}}$$
