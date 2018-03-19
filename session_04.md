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
color image. The intensity of a pixel in a RGB image is represented by three `uint8` integers where these three integers express the mixture of the three base colors - red, green and blue. You can interpret the value of a certain channel as the degree of the color intensity. Because each pixel has three `uint8` values, each RGB pixel can represent $$2^{24}$$ different colors.

The grayscale and the RGB images are only two types of image encodings. There are other image encodings such as YCrCb, HSV, HLS that are widely applied in many other applications. We do not discuss these encodings here.

---

<div align="center">
    <img src="./images/Lenna.png" width="50%">
    <p>Lenna: Goddess of Digital Image Processing.</p>
</div>

---

The above picture is one of the most famous testing images - __Lenna__ (or Lena). The resolution of the image is $$512\times 512$$.  The image has both simple and complex textures, a wide range of colors, nice mixture of detail, shading which do a great job of testing various of image processing algorithms. Lenna is truly goddess of digital image processing. If you would like to read more story of Lenna, please follow [this link](http://www.lenna.org/).

Given a grayscale or RGB image, we can naturally treat the image as the same as a matrix or a 3D tensor. We can then process images with our knowledge of Linear Algebra.

## Image Geometric Transformation

## Simple Image Processing Techniques

## ZCA Whitening

## Image Augmentation in Keras

Data Augmentation 

Keras implements many useful image geometric transformation and normalization methods. The description of the `ImageDataGenerator` is as follows:

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())
```

This generator firstly takes the training features (e.g., $$\mathbf{X}$$) and then computes some statistics,

Here we show an example of using this `ImageDataGenerator`. This generator eliminates the feature-wise mean and normalizes the feature-wise standard deviation. Each image then randomly rotates for at most $$20^{\circ}$$, shifts at most 20% both horizontally and vertically, and performs horizontal flip.

```python
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
```

Noted that this generator is implemented on CPU instead of GPU. Therefore, it takes a long time for preprocessing a huge dataset if you are not careful.

## Exercises
