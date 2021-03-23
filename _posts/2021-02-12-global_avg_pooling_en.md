---
title: "[DL 101] Global Average Pooling"
date: 2021-02-10 12:000 -0400
excerpt: "Alternatives to the Fully Connected Layer(FC layer)"
author : 오승미
categories :
  - deep-learning
tags :
  - deep-learning
  - FC layer
  - GAP

---

# Global Average Pooling

### Alternatives to the Fully Connected Layer(FC layer)

​	In the typical CNN model, we used to extract featues through convolutional layers then add FC layer and softmax layer to the feature map to run classification. FC layer calculates an image's scores for all labels, so we can classify its label by the maximum score. However, FC layer is problematic in the sense that it uses too many parameters, resulting in overfitting easily. In response to this problem, global average pooling(GAP) came out as an alternative.

### How it works?

​	As FC layer, GAP combines vectorized feature map linearly at the last step. However, the difference between them is a transformation matrix. GAP averaged all feature map values.	 

![gap_img](/assets/210212_gap.jpeg)



### How to use this?

​	Pytorch provides **AvgPool2d** layer. We can just add this layer right after running through feature extraction process like below,

```python
nn.Sequential(
          nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(),
          nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(16),
          nn.AvgPool2d(3),
          Lambda(lambda x: x.view(x.size(0), -1)),
    )
```



### Advantages

- interpretable - directly matches feature map to the convolutional structure
- no parameter to optimize; **avoiding overfitting problems!**
- robust to spatial translations of the input



### Reference

- Lin, Min, Qiang Chen, and Shuicheng Yan. "Network in network." *arXiv preprint arXiv:1312.4400* (2013)

- [CSDN](https://blog.csdn.net/yimingsilence/article/details/79227668)
