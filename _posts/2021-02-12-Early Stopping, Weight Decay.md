---
title: "[DL 101] Early Stopping, Weight Decay"
excerpt: "regularization methods: early stopping, weight decay"
date: 2021-02-12 12:000 -0400
author : 오승미
categories :
  - deep-learning
tags :
  - deep-learning
  - regularization
  - early-stopping
  - weight-decay


---

# Early Stopping, Weight Decay



​	Today we are going to briefly talk about two regularization methods: **early stopping, weight decay**.



## 1. Early Stopping

​	Early stopping is a way of arbitrarily specifying large epoch values when training a neural network and stopping training when the model seems to be converged (no siginificant improvement in validation loss). The epoch value is crucial since it is definitely related to underfitting/overfitting problems. With the early stopping method, we can easily deal with this problem. 	

​	However, whether to implement the early stopping is controversial - Andrew Ng does not recommend using it. In fact, not only do you get some good results when you keep training even if there is no noticeable change in the validation error, but you don't know how or how much it will affect the entire process.





## 2. Weight Decay

​	To prevent overfitting, it is necessary to prevent the model from being too complex. Thus, depending on the complexity, we can give the model penality, usually by adding all parameters to the loss function (exactly the squared ver) to prevent excessive parameters. The more parameters there are, the more complex the model becomes.

​	However, adding squared norm of parameters to the loss function causes it to become too large. Thus we can multiply a small number to the sqaure norm to prevent this, called **weight decay**. The overall loss expression is shown below.

```
loss = loss + weight decay parameter * L2 norm of the weights
```

​	We can easily implement weight decay as below,

```python
optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```



## Reference

[1] https://medium.com/analytics-vidhya/deep-learning-basics-weight-decay-3c68eb4344e9

[2] https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab

[3] https://medium.com/zero-equals-false/early-stopping-to-avoid-overfitting-in-neural-network-keras-b68c96ed05d9

[4] https://www.reddit.com/r/MachineLearning/comments/9omr67/discussion_early_stopping_why_not_always/

[5] https://github.com/autonomio/talos/issues/56#issuecomment-413569909
