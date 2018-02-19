---
layout: default
---

Welcome to the second session of Deep Learning in Raspberry Pi.

In this section, we are gonna introduce some core concepts in Machine
Learning (ML). We will dive into two historically very influential learning models: Linear Regression and Logistic Regression. We will also explain Stochastic Gradient Descent (SGD) and its variants. 

## Machine Learning Basics

### What is a "Learning Algorithm"?

A popular definition of learning algorithm is introduced by Tom M. Mitchell in his classical book _Machine Learning_ in 1997:

---

"A computer program is said to learn from

+   experience $$E$$ with respect to
+   some class of tasks $$T$$ and
+   performance measure $$P$$,

if its performance at tasks in $$T$$, as measured by $$P$$, improves with experience $$E$$" (Mitchell, 1997)

---

__NOTE__: This book is really fun to read and introduced many ML algorithms that are very popular back then. It reflects how researchers thought and did in 1980s and 1990s.

#### The task $$T$$

+ __Classification__ specifies which $$k$$ categories some input belongs to. ($$f:\mathbb{R}^{n}\rightarrow\{1,\ldots, K\}$$)

+ __Regression__ predicts a numerical value given some input. ($$f: \mathbf{R}^{n}\rightarrow\mathbf{R}$$)

+ __Transcription__ outputs a sequence of symbols, rather than a category code. (similar to classification, e.g. speech recognition, machine translation, image captioning)

+ __Denoising__ predicts clean samples $$\mathbf{x}$$ from _corrupted_ samples $$\tilde{\mathbf{x}}$$. (estimate $$\Pr(\mathbf{x}\vert\tilde{\mathbf{x}})$$)

And many more types are not listed here.

#### The performance measure $$P$$

+ Measure $$P$$ is usually specific to the task $$T$$ (e.g. accuracy to classification)

+ Batches of unseen _validation_ data is introduced to measure performance.

+ Design measure $$P$$ can be very subtle. It should be effective

#### The experience $$E$$

Experience is what learning algorithms are allowed to have during learning process.

+ Experience is usually an _dataset_, a collection of _examples_.

+ _Unsupervised Learning algorithms_ experience a dataset containing many features, learning useful structure of the dataset (estimate $$\Pr(\mathbf{x})$$).

+ _Supervised Learning algorithms_ experience a dataset containing features, but each example is also associated with a _label_ or _target_ (estimate $$\Pr(\mathbf{y}|\mathbf{x})$$).

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
\hat{\mathbf{v}} =& \mu\mathbf{v}-\alpha\nabla\mathbf{L}(\mathbf{w}+\mu\mathbf{v}) \\
\hat{\mathbf{w}} =& \mathbf{w}+\hat{\mathbf{v}}
\end{aligned}
$$
