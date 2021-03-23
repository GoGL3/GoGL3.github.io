---
title: "[Paper Review] Faster R-CNN"
excerpt: "Fast R-CNN starts from the idea: What if using convolutional feature maps for generating region proposals?"
date: 2021-03-11 02:000 -0400
author : 오승미
use_math: true
tags :
- faster rcnn
- computer vision
categories:
 - computer vision


---

From previous posts, we reviewed R-CNN and Fast R-CNN.

Clearly, Fast R-CNN speeds up the overall process compared to R-CNN but it still uses selective search for region proposals, which is time and cost consuming.

Fast R-CNN starts from the idea:

`What if using convolutional feature maps for generating region proposals?`



**Faster R-CNN** includes region proposal part inside the model instead of the selective search.

Basically Faster R-CNN follows Fast R-CNN structure but *Region Proposal Network* is added between the conv feature map and RoI pooling layer like below.

<img src="/assets/2021-03-11-faster1.png" alt="/assets/2021-03-11-faster1" style="zoom: 50%;" />



## Region Proposal Network

After CNN process, we have a conv feature map that is going to be input to RPN. As you can see in the below picture, **slides a n x n spatial window** (n=3 in this paper) of the feature map to predict region proposals. Note that an anchor is a box in this model and we locate it to the center of the sliding window.

![2021-03-11-faster2](/assets/2021-03-11-faster2.png)



We use $k$ maximum possible anchor boxes for each window and set three anchor box areas of 128<sup>2</sup>, 256<sup>2</sup> and 512<sup>2</sup> pixels and three aspect ratios of 1:1, 1:2, and 2:1, thus nine different anchor boxes in total (k=9). The graph illustrates nine different anchor boxes. Due to a variey in its scales and ratios, we expect it to capture the RoI well.

![2021-03-11-faster4](/assets/2021-03-11-faster4.png)

Thus the regression layer outputs 4k coordinates of k bounding boxes and the classification layer does 2k values that estimate the probabiltity whether an object exists for each anchor box. Since we adopt RPN to predict region proposals, we only focus on 2k classification scores. Reducing the overlapped area between anchors, we implement non-maximum suppression (like bottom-up approach in selective search that we covered [here](https://gogl3.github.io/computer%20vision/fast_rcnn/)) based on classification scores and set the IoU threshold to be 0.7, leaving around only 2000 region proposals for each image.

Then the left procedure is the same as Fast R-CNN; processing RoI pooling where inputs to be the whole image and the predicted bounding boxes (by RPN in Faster R-CNN but selective search in Fast R-CNN) and run through FC layers.

The loss function for this end-to-end learning is
$$
\begin{array}{r}
L\left(\left\{p_{i}\right\},\left\{t_{i}\right\}\right)=\frac{1}{N_{c l s}} \sum_{i} L_{c l s}\left(p_{i}, p_{i}^{*}\right) \\
+\lambda \frac{1}{N_{r e g}} \sum_{i} p_{i}^{*} L_{r e g}\left(t_{i}, t_{i}^{*}\right)
\end{array}
$$
where i is the index of an anchor in a mini-batch and pi is the predicted probability of anchor i being an object. If $p_i$ is 1, the anchor is positive and is 0 otherwise. $t_i$ denotes four coordinates for the predicted bounding box.





## Reference

- [https://arxiv.org/pdf/1506.01497.pdf](https://arxiv.org/pdf/1506.01497.pdf)
- [https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8)
