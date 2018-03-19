---
layout: default
---

In this session, we focus on introducing common concepts and techniques in __Digital Image Processing__. We also would like to discuss how you can apply these techniques to Machine Learning tasks.

## Digital Image Representation

Before giving intuitive examples of _digital images_ or simply _images_, we would like to formally define the concept of digital image and _digital image processing_. One accurate definition is from the book _Digital Image Processing (2nd Edition)_ by Rafael C. Gonzalez and Richard E. Woods.

---

An image may be defined as a two-dimensional function, $$f(x,y)$$, where $$x$$ and $$y$$ are _spatial_ (plane) coordinates, and the amplitude of $$f$$ at any pair of coordinates $$(x,y)$$ is called the _intensity_ or _gray level_ of the image at that point. When $$x$$, $$y$$, and the amplitude values of $$f$$ are all finite, discrete quantities, we call the image a _digital image_. The field of _digital image processing_ refers to processing digital images by means of
a digital computer.

---

__Remark__: Although this book was published over a decade ago, the book presents a comprehensive introduction that both beginners and experts can enjoy reading.

With the above definition, we can simply define an image as a $$n\times m\times k$$ (`height x width x color_channels`) array where the image consists of $$n\times m$$ _pixels_ and $$n, m, k\in \mathbb{N}^{+}$$. A pixel is a point $$(x,y)$$ in a image. Typically, the intensity of a pixel is denoted as one or an array of `uint8` values. `uint8` is unsigned 8-bit integer that has the range $$[0, 255]$$ (0 is no color, 255 is full color). If there is only one color channel
($$k=1$$), the image is typically stored as a _grayscale_ image. The intensity of a pixel is then represented by one `uint8` value where 0 is black and 255 is white. If there are three color channels ($$k=3$$), we define that the first, the second and the third channel are the <font color="red">red</font> channel, the <font color="green">green</font> channel and the <font color="blue">blue</font> channel respectively. We hence refer this type of image to as _RGB_
color image.

__Remark__: In this module, we assume that we always use either grayscale image or RGB image.

---

<div align="center">
    <img src="./images/Lenna.png">
    <p>Lenna: Goddess of Digital Image Processing.</p>
</div>

---

## Image Geometric Transformation

## Simple Image Processing Techniques

## PCA and ZCA Whitening

## Exercises
