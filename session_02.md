---
layout: default
---

## Linear Regression

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
