---
layout: default
---

## Linear Regression

```python
x = Input((10,), name="input_layer"
y = Linear(1, name="linear layer")
model = Model(x, y)
```

## Logistic Regression

```python
x = Input((10,), name="input_layer"
y = Linear(1, name="linear layer")
y = Activation("softmax")
model = Model(x, y)
```

## Stochastic Gradient Descent and its variants

$$\boldmath{\theta}^{\star}=\boldmath{\theta}-\alpha\frac{\partial\mathcal{L}}{\partial \boldmath{\theta}}$$
