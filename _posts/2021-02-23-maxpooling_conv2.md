---
title: "[DL 101] Maxpooling? Convolutions?"
excerpt: "Can we replace maxpooling with convolutional layers?"
date: 2021-02-22 15:000 -0400
author : 오승미
categories :
  - deep-learning
tags :
  - deep-learning
  - Maxpooling
  - Convolution
---



You might be familiar with the Max Pooling which takes maximum values to down sample the feature map; lesser parameters for being robust to changes.

However, maxpooling can be replaced with convolutional layers. Why? cause some feature information can be lost when running through maxpooling layers. The [medium article](https://medium.com/@duanenielsen/deep-learning-cage-match-max-pooling-vs-convolutions-e42581387cb9) shows an example of image reconstruction using VAE to compare models with a maxpooling layer and a convolutional layer. As you can see in that article, the convolutional pooling outperforms the one with the maxpooling.



Addition to this, [STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET](https://arxiv.org/pdf/1412.6806.pdf ) says that

> We find that max-pooling can simply be replaced by a convolutional layer with increased stride **without loss in accuracy** on several image recognition benchmarks.

> when pooling is replaced by an additional convolution layer with stride r = 2 performance stabilizes and **even improves** on the base model



## Reference

- https://medium.com/@duanenielsen/deep-learning-cage-match-max-pooling-vs-convolutions-e42581387cb9

- https://github.com/DuaneNielsen/maxpoolvsconv

- https://arxiv.org/pdf/1412.6806.pdf
- https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling
