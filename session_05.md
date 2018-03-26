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

## Corner Detection

## Keypoints Detection

## Feature Matching

## Face Detection

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

## Tips and Tricks in Machine Learning

## Closing Remarks

In the last five sessions, we learned how to use Keras for symbolic computation
and solving Machine Learning problems. We also learned how to use
OpenCV, numpy, scikit-image to process image data.
You might be overwhelmed by how many softwares there are in order to solve
some particular problems. You probably wonders if this is enough for
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
