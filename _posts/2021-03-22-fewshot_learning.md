---
title: "[CV] Few Shot Learning"
excerpt: "Intro to Few Shot Learning"
date: 2021-03-22 06:000 -0400
author : 오승미
use_math: true
tags :
- computer vision
- few shot learning

categories:
- computer vision
- deep-learning

---





meta learning: learn to learn

 knows similarity and differences between objects during training.

aim: learn a similarity function

support set : small set of samples, too small for training

- for human, make prediction based on four train dataset is easy
- on the other hand, not know
- but we can train how much two input images are alike. 
- find the query most similar to the support set
- != training set. training set is big enough for training
- Although the otter image is not included in training set, if it is in support set, the model can tell that the query image looks similar to the otter

query : the unknown animal 



Supervised learning vs. few shot learning

- test samples are from known classes
- f: query samples are from unknown classes
- f: need to provide more information. a set of cars (support set)

k-way: the support set has k classses

n-shot : every class has n samples

N은 10개 이하, K를 1개 또는 5개로 설정

쿼리 데이터는 범주당 15개



## metric based

## steps:

1. learn a simliarity function from large scaled image set

2. then apply the sim func for prediction
   - compare the query with every sample in the support set
   - find the sample with the highest similarity score

Dataset: Omniglot

- 50 Alphabets * # characters * 20 handwritten letters

Dataset: Mini-ImageNet



 ### ex. Siamese Networks

training data = positive samples + negative samples

pos: (tiger image1, tiger image 2, 1) 

Neg: (car image1, elephant image2, 0) ; two are different



The idea is to find model hyper-parameters and parameters such that it will be easy to adapt to a new task without over-fitting to the few shots available.



## Few-shot learning with pretrained CNN

- pretrain a CNN for feature extraction
- can be pretrained using standard supervised learning or Siamese
- mean of feature vectors of a class then take a norm, like mu1
- for query image, take the same step and get a unit vector q
- compare q with mu1, mu2, mu3 on 2-dim space
- p = softmax(Mq)

![Screen Shot 2021-03-21 at 9.33.27 PM](/Users/seungmi/Library/Application Support/typora-user-images/Screen Shot 2021-03-21 at 9.33.27 PM.png) 

where M = [mu1, mu2, mu3]'

 



## Reference

- https://www.kakaobrain.com/blog/106
- https://www.youtube.com/watch?v=hE7eGew4eeg&t=0s&ab_channel=ShusenWang
- https://www.youtube.com/watch?v=4S-XDefSjTM&ab_channel=ShusenWang
- https://www.youtube.com/watch?v=U6uFOIURcD0&ab_channel=ShusenWang
- https://towardsdatascience.com/building-a-one-shot-learning-network-with-pytorch-d1c3a5fafa4a
- http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
