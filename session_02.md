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
f(\mathbf{x}) =& \frac{1}{1-\exp(-\mathbf{W}^{\top}\mathbf{x})} \\
\Pr(y=1|\mathbf{x}) =& f(\mathbf{x}) \\
\Pr(y=0|\mathbf{x}) =& 1-f(\mathbf{x})
\end{aligned}
$$

$$
\mathcal{L}(\mathcal{X}, \mathbf{y}|\mathbf{W}) = -\frac{1}{N}\sum_{i}\left(y^{(i)} \log(\Pr(y=1|\mathbf{x}^{(i)}))+(1-y^{(i)})\log(\Pr(y=0|\mathbf{x}^{(i)}))\right)
$$

$$
\mathbf{W}^{\star}=\arg\min_{\mathbf{W}}\mathcal{L}(\mathcal{X}, \mathbf{y})
$$

$$
\text{softmax}(\mathbf{x})=\Pr(y=k|\mathbf{x}, \mathbf{W}) = \frac{\exp(\mathbf{W}^{k\top}\mathbf{x})}{\sum_{j=1}^{K}\exp(\mathbf{W}^{(j)\top}\mathbf{x})}
$$

$$
\mathcal{L}(\mathcal{X}, \mathbf{y}|\mathbf{W}) = -\frac{1}{N}\sum_{i}\sum_{k=1}^{K}\mathbf{1}\{y^{(i)}=k\}\log\Pr(y^{(i)}=k|\mathbf{x}^{(i)}, \mathbf{W})
$$

$$
\mathbf{W}^{\star}=\arg\min_{\mathbf{W}}\mathcal{L}(\mathcal{X}, \mathbf{y})
$$

```python
x = Input((10,), name="input_layer")
y = Linear(1, name="linear layer")
y = Activation("softmax")
model = Model(x, y)
```

## Stochastic Gradient Descent and its variants

$$\mathbf{w}^{\star}=\mathbf{w}-\alpha\frac{\partial\mathcal{L}}{\partial \mathbf{w}}$$

$$
\begin{aligned}
\hat{\mathbf{v}}=&\mu\mathbf{v}-\alpha\nabla\mathcal{L}(\mathbf{w}) \\
\hat{\mathbf{w}}=&w+\hat{\mathbf{v}}
\end{aligned}
$$

$$
\begin{aligned}
\hat{\mathbf{v}} = \mu\mathbf{v}-\alpha\nabla\mathbf{L}(\mathbf{w}+\mu\mathbf{v}) \\
\hat{\mathbf{w}} = \mathbf{w}+\hat{\mathbf{v}}
\end{aligned}
$$
