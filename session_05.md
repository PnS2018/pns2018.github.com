---
layout: default
---

In this session, we are going to discuss some advanced topics in
Computer Vision without explaining too much mathematical details.
These are essential tools that you can use everyday in your
projects.

We are also going to discuss some technical topics including how to
interact with the camera on Raspberry Pi, how to develop a proper
Python project, and some additional tips and tricks for
solving Machine Learning problems in general.

## Work with Pi Camera

We have equipped the Raspberry Pi Camera on the Raspberry Pi
so that you can get the video feed from the device.

In order to stream the frames from the Pi Camera, we need to
use a new Python package called `picamera[array]`.
The reason why we do not use OpenCV here is because one need to
solve tedious driver issues.

The following example demonstrate how you can get a stream of frames
from the Pi Camera ([Script citation](https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/)):

```python
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  # grab the raw NumPy array representing the image, then initialize the timestamp
  # and occupied/unoccupied text
  image = frame.array

  # Display the resulting frame
  cv2.imshow('frame', image)
  # the loop breaks at pressing `q`
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # clear the stream in preparation for the next frame
  rawCapture.truncate(0)
```

A lot more examples can be found at the [documentation](https://picamera.readthedocs.io/en/release-1.13/) of the library.

## Work with Webcam

OpenCV provides a good interface for streaming data from
a video or a camera. The `VideoCapture` API can accept
a string that indicates the path of the video or
an integer that indicates a camera. Commonly, we use
the integer 0 (corresponding to `CAP_ANY`) so that the OpenCV
auto-detects the camera devices available.

Note that we only discuss a single camera setup, for accessing
multiple cameras, you need special treatments such as
camera synchronization.

Here we demonstrate an example that shows the basic usage
of the `VideoCapture` API:

```python
import numpy as np
import cv2

# open the camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    # the loop breaks at pressing `q`
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```

OpenCV's APIs on reading videos are perhaps the most easy-to-use
ones on the market. But of course, there are different solutions:

+ `scikit-video` supports ranges of video related functions from reading and writing video files to motion estimation functions.
+ `moviepy` is a Python package for video editing. This package works quite well on some basic editing functions (such as cuts, concatenations, title insertion) and writing video (such as in GIF) with intuitive APIs

We encourage you to explore these options, but so far the OpenCV is perhaps
the most widely adopted solution for processing videos.

__Remark__: In the domain of Deep Learning, we rarely use the image
that has the resolution higher than `250x250`. Usually, we will
rescale and subsample them so that we would not have to process
every details of the picture. We also suggest that for your projects,
you can cut the unnecessary parts, rescale the image or select only
the regions of interests.

## Feature Engineering

In classic Computer Vision, __Feature Engineering__ is an important step
before further analysis such as detection and recognition.
Commonly, this process selects/extracts useful information from the raw
inputs and passes to the next stage so that the later processes can
both focus on relevant details and reduce the computing overhead.

For example, when we want to identify a person from a database,
we can simply compare every picture available in the database
with the photo at the raw pixel level. This process may be effective
if these pictures records the front face and are aligned perfectly
and the lighting conditions are roughly the same.
Besides the limitation on the input data, this algorithm
also generates huge amount of computing overhead that it may not
be feasible for running on a reasonable computer. Instead,
we can first extract the relevant features that identify the
person of interest, and then search the corresponding one
in the database. This alternative solution would boost the performance
greatly since you could identify the person with these selected features,
and these features are designed to be robust in different environments.

Note that in recent years, conventional Feature Engineering is
largely replaced by Deep Learning systems where these system
integrates the Feature Engineering step into the architecture itself.
Instead of relying on careful designed features, the features learned
by these DL systems are proven to be more effective and robust to
the changes in the dataset.

In the next sections, we introduce some common feature engineering
techniques and show how we can use them for solving
Computer Vision tasks.

## Corner Detection

Corner Detection is one of the classical procedure of Computer Vision
preprocessing due to the reason that the corners represent
a large variation in intensity in many directions.
And this can be very attractive in our above photo matching example.
If we can find the near identical corner features from two
different photos, we can roughly say that they share similar content
in the pictures (of course in reality this is not enough).

We give a OpenCV example as follows.

```python
import cv2
import numpy as np

img = cv2.imread("Lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

dst = cv2.dilate(dst, None)

img[dst > 0.01*dst.max()] = [0, 0, 255]

cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
```

The above code uses _Harris Corner Detector_. OpenCV implements this
method in `cornerHarris` API. This API receives
four arguments:

+ `img`: input image, it should be grayscale and `float32` type.
+ `blockSize`: it is the size of the neighbourhood considered for corner detection
+ `ksize`: Aperture parameter of Sobel (from last session) derivative used.
+ `k`: Harries detector free parameter in the equation.

You will need to tune this corner detector for different image so that
you can get optimal results.

## Keypoints Detection



```python
import cv2

img = cv2.imread('Lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# for opencv 3.x and above
# you will need to run the following code to install
# pip install opencv-contrib-python -U
sift = cv2.xfeatures2d.SIFT_create()
# for raspberry pi
sift = cv2.SIFT()

kp = sift.detect(gray, None)

# for opencv 3.x and above
cv2.drawKeypoints(gray, kp, img)
# for Raspberry Pi
img = cv2.drawKeypoints(gray, kp)

cv2.imshow('dst', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

SURF is faster than SIFT

## Feature Matching

TODO: to prepare testing images for feature matching

## Face Detection

In order to proceed, you will need to install the latest version of
the `pnslib`. If you don't have it, first clone the project to any
directory:

```
$ git clone https://github.com/PnS2018/pnslib
$ cd pnslib
```

If you've installed the library, first, you need to pull the
latest changes:

```
$ git pull origin master
```

Install the package via

```
$ python setup.py develop
```

```python
import cv2
from pnslib import utils

# read image
img = cv2.imread("Lenna.png")

# load face cascade and eye cascade
face_cascade = cv2.CascadeClassifier(
    utils.get_haarcascade_path('haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(
    utils.get_haarcascade_path('haarcascade_eye.xml'))

# search face
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Available Haar Cascades in OpenCV

```
1. haarcascade_eye.xml
2. haarcascade_eye_tree_eyeglasses.xml
3. haarcascade_frontalcatface.xml
4. haarcascade_frontalcatface_extended.xml
5. haarcascade_frontalface_alt.xml
6. haarcascade_frontalface_alt2.xml
7. haarcascade_frontalface_alt_tree.xml
8. haarcascade_frontalface_default.xml
9. haarcascade_fullbody.xml
10. haarcascade_lefteye_2splits.xml
11. haarcascade_licence_plate_rus_16stages.xml
12. haarcascade_lowerbody.xml
13. haarcascade_profileface.xml
14. haarcascade_righteye_2splits.xml
15. haarcascade_russian_plate_number.xml
16. haarcascade_smile.xml
17. haarcascade_upperbody.xml
```

## How to Develop a Python Project

You will be working on your projects from the next session on. Working on a
deep learning project involves lots of data, experiments and analyzing the
results. So, it is imperative that you maintain a good structure in your project
folder so you can handle your work in a smooth fashion. Here are a few tips
that will help you in this regard. Note that, the points mentioned below are
just helpful guidelines, and you free to use it, tweak it or follow a
completely different folder structure as per your convenience.

Before you start a project, there are a few things (not an exhaustive list by any means) that you have to do in general.

1. Implement a model architecture
2. Train the model
3. Test the model
4. Take care of the data
5. Take care of any hyper parameter tuning (optional)
6. Keep track of the results
7. Replicate any experiment

So your folder structure should take care of these requirements.

#### Folder structure
Initialize a github project on the pns2018 github repository, and clone it to your working machine. The folder structure can be organized as follows.

1. __README__ file: A README file with some basic information about the project, like the required dependencies, setting up the project and replicating an experiment, links to the dataset, is really useful both for you and anyone you are sharing your project with.
2. __data__ folder: A data folder to host the data is useful if it is kept separate. Given that some of your projects involve data collection, your data generation scripts can also be hosted in this directory.
3. __models__ folder: You can put your model implementations in this folder.
4. __main.py__ file: The main file is the main training script for the project. Ideally you would want to run this script with some parameters of choice to train a required model.
5. __tests__ file: You can put your testing files in this folder.
6. __results__ folder: You can save all your results in your project in this folder, including saved models, graphs of your training progress, other results when testing your models that you want to pull out later.




## Tips and Tricks in Machine Learning

## Closing Remarks

In the last five sessions, we learned how to use Keras for symbolic computation
and solving Machine Learning problems. We also learned how to use
OpenCV, numpy, scikit-image to process image data.
You might be overwhelmed by how many softwares there are in order to solve
some particular problems. You probably wonder if this is enough for
dealing with practical issues in future projects.

The answer is a hard NO. Keras is might be the best way of prototyping
new ideas. However, it is not the best software to scale,
it is not the fastest ML software on the market, and researchers from
different domains have different flavors of choosing and using software.
In [this page](./dl-res.html), we listed some influential
softwares that are currently being employed in academia and industry.

The best way of learning a programming language/library is arguably
to do a project. In the following weeks, you will need to
choose a project that uses the knowledge you learned in the past
five weeks. Furthermore, everything has to be fit into a Raspberry Pi.
