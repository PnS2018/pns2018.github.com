---
layout: default
---

Welcome to the second session of Deep Learning in Raspberry Pi.

In this section, we are going to introduce some core concepts in Machine
Learning (ML). We will dive into two historically very influential learning models: Linear Regression and Logistic Regression. We will also discuss Stochastic Gradient Descent (SGD) and its variants. 

## What is a "Learning Algorithm"?

A broadly adopted definition of _learning algorithm_ is given by Tom M. Mitchell in his classical book _Machine Learning_ in 1997:

---

"A computer program is said to __learn__ from

+   experience $$E$$ with respect to
+   some class of tasks $$T$$ and
+   performance measure $$P$$,

if its performance at tasks in $$T$$, as measured by $$P$$, improves with experience $$E$$" (Mitchell, 1997)

---

__NOTE__: This book is really fun to read and introduced many ML algorithms that are very popular back then. It reflects how researchers thought and did in 1980s and 1990s.

Many popular machine learning textbooks have in-depth discussions of this
definition (Mitchell, 1997; Murphy, 2012; Goodfellow et al., 2016). Here we give simple examples for them:

### The task $$T$$

+ __Classification__ specifies which $$k$$ categories some input belongs to. ($$f:\mathbb{R}^{n}\rightarrow\{1,\ldots, K\}$$)

+ __Regression__ predicts a numerical value given some input. ($$f: \mathbf{R}^{n}\rightarrow\mathbf{R}$$)

+ __Transcription__ outputs a sequence of symbols, rather than a category code. (similar to classification, e.g. speech recognition, machine translation, image captioning)

+ __Denoising__ predicts clean samples $$\mathbf{x}$$ from _corrupted_ samples $$\tilde{\mathbf{x}}$$. (estimate $$\Pr(\mathbf{x}\vert\tilde{\mathbf{x}})$$)

And many more types are not listed here.

### The performance measure $$P$$

+ Measure $$P$$ is usually specific to the task $$T$$ (e.g. accuracy to classification)

+ Batches of unseen _validation_ data is introduced to measure performance.

+ Design measure $$P$$ can be very subtle. It should be effective

### The experience $$E$$

Experience is what learning algorithms are allowed to have during learning process.

+ Experience is usually an _dataset_, a collection of _examples_.

+ _Unsupervised Learning algorithms_ experience a dataset containing many features, learning useful structure of the dataset (estimate $$\Pr(\mathbf{x})$$).

+ _Supervised Learning algorithms_ experience a dataset containing features, but each example is also associated with a _label_ or _target_ (estimate $$\Pr(\mathbf{y}\vert\mathbf{x})$$).


Mathematically, this computer program with respect to the learning task $$T$$ can be defined
as a hypothesis function that takes an input $$\mathbf{x}$$ and transforms it to
an output $$\mathbf{y}$$.

$$\mathbf{y}=f(\mathbf{x}; \mathbf{W})$$

The function may be parameterized by a group of parameters $$\mathbf{W}$$.
Note that $$\mathbf{W}$$ includes both trainable and non-trainable parameters.
All the DNN architectures discussed in this module can be formulated in this paradigm.


## Linear Regression

$$
\begin{aligned}
y=&w_{1}\cdot x_{1}+w_{2}\cdot x_{2}+\ldots+w_{i}\cdot x_{i}+\ldots+w_{n}+x_{n}+b \\
=&\sum_{i=1}^{n}w_{i}\cdot x_{i}+b
\end{aligned}
$$


```python
x = Input((10,), name="input layer")
y = Linear(1, name="linear layer")
model = Model(x, y)
```

## Generalization, Capacity, Overfitting, Underfitting

+   __Generalization__ abaility to perform well on previously unobserved inputs.
+   __Capacity__ abaility to fit a wide varity of functions.
+   __Overfitting__ occurs when the gap between training error and test error is too large
+   __Underfitting__ occurs when the model is not able to obtain a sufficiently low error value on the training set.

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

## Hyperparameters, Validation sets

+   __Hyperparameters__ controls the behavior of the learning algorithm. Usually choose empirically.

+   __Validation sets__ is a subset of data used to guide the selection of hyperparameters. (split from training dataset, usually 20%)

## No Free Lunch Theorem, Curse of Dimensionality

---

The no free lunch theorm for machine learning (Wolpert, 1996) states that, average ove rall possible data generating distributions, every classification algorithm has the same error rate when classifying previous unobserved points. In some sence, no ML algorithm is universally any better than any other.

---

__Seek solution for some relevant distributions, NOT universal distribution.__


---

Many machine leanring problems become exceedingly difficult when the number of dimensions in the data is high. The phenomenon is known as the _curse of dimensionality_. Of particualr concern is that the number of possible distinct configurations of the variables of interest increases __exponentially__ as the dimensionality increases.

---
