---
title: "[DL 101] Object Recognition Terminology"
excerpt: "What is object localization, object detection, semantic segmentation and instance segmentation?"
date: 2021-02-11 13:000 -0400
author : 오승미
categories :
  - deep-learning
tags :
  - deep-learning
  - object-recognition
  - object-detection
  - segmentation


---

# Object Recognition Terminology

This is based on [CS231n note](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf) and [stackoverflow post](https://stackoverflow.com/questions/33947823/what-is-semantic-segmentation-compared-to-segmentation-and-scene-labeling%20).

![Alt text](/assets/210212_object_local.png)

## 1. Object Localization

​	It is used to locate an object from a given image, as shown in the second picture. It is called **bounding box**, consisting of the coordinates of the square-shaped space (x-axis and y-axis of its center, height, width) that fits the object compactly, such as the red square above.

## 2. Object Detection

​	It is almost the same as the object localization, but object detection can fetch each bounding box for multiple objects like the third picture above.

## 3. Semantic Segmentation

​	It does not distinguish instances of the same class. I mean, in the second figure below, the all chairs in the picture are labeled as blue -  no difference between chairs (each chair = an instance).

The class is labeled per pixel because it focuses only on the pixel of the image and determines which class the pixel belongs to. Let's just say that there is an image of 512*512, class 1 is a person, 2 is a motorcycle, and 3 is a car. Then the image is labeled like

000000000000

0000001111100

...

where 0 denotes background.

![enter image description here](https://i.stack.imgur.com/mPFUo.jpg)

## 4. Instance Segmentation

​	Further to semantic segmentation, even the same class is expressed in different colors to distinguish the instance. In the fourth picture right above, each chair is distinguished by different colors.
