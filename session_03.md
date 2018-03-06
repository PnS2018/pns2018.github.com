---
layout: default
---

## Deep Learning: Another Jon Snow in CS Naming Logic

__Deep Learning__ (DL) is a terrible name. First of all, no one has defined a concrete definition of "deep" that everyone agrees. In the influential review article _Deep Learning in Neural Networks: An Overview_, Schmidhuber says:

---

At which problem depth does _Shallow Learning_ end, and _Deep Learning_ begin? Discussions with DL experts have not yet yielded a conclusive response to this question. Instead of committing myself to a precise answer, let me just define for the purposes of this overview: problems of depth $$>10$$ require Very Deep Learning.

---

__Remark__: We encourage you to check out this fascinating review of Deep Learning. We also strongly encourage you to check out the author Jurgen Schmidhuber who is one of the founding father of deep learning.

Instead of the term "deep", a term that might be more suitable is "hierarchical". However, modern (in last 2 years) deep learning models cannot simply be characterized by "hierarchical" anymore (e.g, Neural Turning Machine models).

The second part of the name "Learning" is also a misleading word. If you look at the models we present in this module closely, there is nothing in the models that is associated with "learning" in the biological sense. The entire "learning" process is carried out by some powerful optimization algorithms (we called them "training" algorithms).

I (Yuhuang) personally believe that this is another example where computer scientists gave a terrible yet catchy name (the first example is "Computer Science" itself, check [here](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-001-structure-and-interpretation-of-computer-programs-spring-2005/video-lectures/1a-overview-and-introduction-to-lisp/) for reason). Yann Lecun, one of the founding father of Deep Learning, proposed to rename "Deep Learning" to "Differentiable
Programming" in a recent [Facebook post](https://www.facebook.com/yann.lecun/posts/10155003011462143). I think this is by far the most accurate definition. First, all modern deep learning models up to date are differentiable. And the optimization algorithms finds a set of optimal parameters that "program" the model to exhibit some desired behaviors.

We have to remember that DL consists of many extremely useful and powerful tools. However, DL alone is not AI, or Machine Learning.
