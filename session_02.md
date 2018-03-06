---
layout: default
---

Welcome to the second session of Deep Learning in Raspberry Pi.

In this section, we are going to discuss some core concepts in Machine
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

__Remark__: This book is really fun to read and introduced many ML algorithms that were very popular back then. It reflects how researchers thought and did in 1980s and 1990s.

Many popular machine learning textbooks have in-depth discussions of this
definition (Mitchell, 1997; Murphy, 2012; Goodfellow et al., 2016).

### The task $$T$$

+ __Classification__ specifies which $$k$$ categories some input belongs to. ($$f:\mathbb{R}^{n}\rightarrow\{1,\ldots, K\}$$)

+ __Regression__ predicts a numerical value given some input. ($$f: \mathbb{R}^{n}\rightarrow\mathbb{R}$$)

+ __Transcription__ outputs a sequence of symbols, rather than a category code. (similar to classification, e.g. speech recognition, machine translation, image captioning)

+ __Denoising__ predicts clean samples $$\mathbf{x}$$ from _corrupted_ samples $$\tilde{\mathbf{x}}$$. (estimate $$\Pr(\mathbf{x}\vert\tilde{\mathbf{x}})$$)

And many more types are not listed here.

### The performance measure $$P$$

+ Measure $$P$$ is usually specific to the task $$T$$ (e.g. accuracy to classification).

+ Batches of unseen _validation_ data is introduced to measure performance.

+ Design measure $$P$$ can be very subtle. It should be effective.

### The experience $$E$$

Experience is what learning algorithms are allowed to have during learning process.

+ Experience is usually an _dataset_, a collection of _examples_.

+ _Unsupervised Learning algorithms_ experience a dataset containing many features, learning useful structure of the dataset (estimate $$\Pr(\mathbf{x})$$).

+ _Supervised Learning algorithms_ experience a dataset containing features, but each example is also associated with a _label_ or _target_ (estimate $$\Pr(\mathbf{y}\vert\mathbf{x})$$).


### Hypothesis function

Mathematically, this computer program with respect to the learning task $$T$$ can be defined
as a hypothesis function that takes an input $$\mathbf{x}$$ and transforms it to
an output $$\mathbf{y}$$.

$$\mathbf{y}=f(\mathbf{x}; \mathbf{W})$$

The function may be parameterized by a group of parameters $$\mathbf{W}$$.
Note that $$\mathbf{W}$$ includes both trainable and non-trainable parameters.
All the DNN architectures discussed in this module can be formulated in this paradigm.

Strictly speaking, the hypothesis function defines a large family of functions that could be the solution to the task $$T$$. At the end of training, the hypothesis function is expected to be parameterized by a set of optimal parameters $$\mathbf{W}^{\star}$$ that yields the highest performance according to the performance measure $$P$$ of the given task.

### The Cost Function

A cost function $$J$$ is selected according to the objective(s) of the hypothesis function in which it defines the constraints. The cost function is minimized during the training so that the hypothesis function can be optimized and exhibits the desired behaviors (e.g., classify images, predict houshold value, text-to-speech). The cost function reflects the performance measure $$P$$ directly or indirectly. In most cases, the performance of a learning algorithm gets higher
when the cost function $$J$$ becomes lower.

When the cost function is differentiable (such as in DNNs presented in this module), a class of _Gradient-Based Optimization_ algorithms can be applied to minimize the cost function $$J$$. Thanks to specialized hardware such as GPUs and TPUs, these algorithms can be computed very efficiently.

Particularly, Gradient Descent (Cauchy, 1847) and its variants, such as
RMSprop (Tieleman & Hinton, 2012), Adagrad (Duchi et al., 2011), Adadelta
(Zeiler, 2012), Adam (Kingma & Ba, 2014) are surprisingly good at training
Deep Learning models and have dominated the development of training algorithms. Software libraries such as `Theano` (Theano Development Team, 2016)
and `TensorFlow` (Abadi et al., 2015) have automated the process of computing the gradient (the most difficult part of applying gradient descent) using
a symbolic computation graph. This automation enables the researchers to
design and train arbitrary learning models.

__Remark__: in this module, we use the term "cost function", "objective function" and "loss function" interchangeably. Usually, the loss function is denoted as $$\mathcal{L}$$.

We will revisit this topic at the end of this session. In next sections, we will look closely into __Linear Regression__ (Regression) and __Logistic Regression__ (Classification).

## Linear Regression

Regression is a task of Supervised Learning. The goal is to take a input vector $$\mathbf{x}\in\mathbb{R}^{n}$$ (a.k.a, features) and predict a target value $$y\in\mathbb{R}$$. In this section, we will learn how to implement _Linear Regression_.

As the name suggested, Linear Regression has a hypothesis function that is a linear function. The goal is to find a linear relationship between the input features and the target value:

$$
\begin{aligned}
y^{(i)}=f(\mathbf{x}^{(i)};\{\mathbf{W}, b\})=&w_{1}\cdot x_{1}+w_{2}\cdot x_{2}+\ldots+w_{i}\cdot x_{i}+\ldots+w_{n}\cdot x_{n}+b \\
=&\sum_{i=1}^{n}w_{i}\cdot x_{i}+b = \mathbf{W}^{\top}\mathbf{x}^{(i)}+b
\end{aligned}
$$

Note that $$\{\mathbf{x}^{(i)}, y^{(i)}\}$$ is the $$i$$-th sample in the dataset $$\{\mathcal{X}, \mathbf{y}\}$$ that has $$N$$ data points. The parameters $$\theta=\{\mathbf{W}, b\}$$ consists of weights $$\mathbf{W}$$ and a bias $$b$$.

Suppose that the target value is a scalar (a.k.a $$y^{(i)}\in\mathbb{R}$$), we can easily define such model in Keras:

```python
x = Input((10,), name="input layer")  # the input feature has 10 values
y = Linear(1, name="linear layer")  # implement linear function
model = Model(x, y)  # compile the hypothesis function
```

To find a linear relationship that has $$y^{(i)}\approx f(\mathbf{x}^{(i)};\theta)$$, we need to find a set of parameters $$\theta^{\star}$$ from the parameter space $$\theta$$ where the optimized function $$f(\mathbf{x};\theta^{\star})$$ generate least error as possible. Suppose we have a cost function $$J$$ that measures the error made by the hypothesis function, our goal can be formulated into:

$$
\theta^{\star}=\arg\min_{\theta}J(\theta)
$$

For Linear Regression, one possible formulation of the cost function is Mean-Squared Error (MSE), this cost function measures the mean error caused by each data sample:

$$
J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\left(y^{(i)}-f(\mathbf{x}^{(i)};\theta)\right)^{2}
$$

By minimizing this cost function via training algorithm such as Stochastic Gradient Descent (SGD), we hope that the trained model $$f(\mathbf{x}; \theta^{\star})$$ can perform well on unseen examples in the testing dataset.

__Remark__: there are other cost functions for regression tasks, such as Mean Absolute Error (MAE) and Root-Mean-Square Error (RMSE). Interested readers are encouraged to find out what they are.

__Remark__: Linear Regression is a class of learning model that are extensively studied in history.

__Remark__: The math in this module choses to use a column-vector based system, which means each vector is assumed to be a column vector. This convention is also applied by many books and tutorials. However, in practice, most `ndarray` packages use the row-vector based system because the first dimension of a multi-dimensional array is for row. For example,

```python
A = np.array([1, 2, 3, 4, 5])
```

The array `A` is actually a row vector. We assume that the readers know this fact and can modify the code accordingly.

## Logistic Regression

In this section, we discuss the solution to another Supervised Learning task - _Binary Classification_. Instead of predicting continuous values (e.g., how many pairs of shoes you have), we will wish to make a decision whether the input feature $$\mathbf{x}$$ belongs to some category. And in the case of Binary Classification, we have only two classes (e.g., to be or not to be, shoe or skirt). And _Logistic Regression_ is a simple learning algorithm that solves this kind of tasks.

Suppose our input feature $$\mathbf{x}$$ is a $$n$$-dimensional vector and the output class label $$y\in\{0, 1\}$$ (0 and 1 are abstract labels, we can associate meanings for these labels, such as 0 is shoe and 1 is skirt). The Logistic Regression constructs a hypothesis function that assign the probability that $$\mathbf{x}$$ belongs to the class $$y=1$$. Specifically, the Logistic Regression uses the "logistic function":

$$
\sigma(\mathbf{x}; \theta) =& \frac{1}{1+\exp(-(\mathbf{W}^{\top}\mathbf{x}+b))}
$$

Commonly, we use the symbol $$\sigma(\cdot)$$ to represent the logistic function. Furthermore, $$\sigma(\cdot)$$ is often called the "sigmoid" function as well. The logistic function has a nice property where it can map the input $$\mathbf{W}^{\top}\mathbf{x}+b$$ into the range $$(0, 1)$$ so that we can interpret the output of this function as probability:

$$
\begin{aligned}
\Pr(y=1|\mathbf{x}) =& \sigma(\mathbf{x}; \theta) \\
\Pr(y=0|\mathbf{x}) =& 1-\sigma(\mathbf{x}; \theta)
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
y = Activation("sigmoid")
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

## Generalization, Capacity, Overfitting, Underfitting

+   __Generalization__ abaility to perform well on previously unobserved inputs.
+   __Capacity__ abaility to fit a wide varity of functions.
+   __Overfitting__ occurs when the gap between training error and test error is too large
+   __Underfitting__ occurs when the model is not able to obtain a sufficiently low error value on the training set.

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


## Exercises
