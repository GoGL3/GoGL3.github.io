---
title: "[DL 101] Early Stopping, Weight Decay"
date: 2021-02-12 12:000 -0400
author : 오승미
categories :
  - Deep Learning
  - Regularization
  - Early Stopping
  - Weight Decay


---

# Early Stopping, Weight Decay



​	Today we are going to briefly talk about two regularization methods: **early stopping, weight decay**.



## 1. Early Stopping

​	Neural Network를 훈련할 때 임의로 큰 epoch 값을 지정하고, 모델이 어느정도 converge한다고 판단될 때 훈련을 중단시키는 방법이다. Epoch 값 또한 underfit/overfit와 연관이 있기 때문에 이를 신중히 판단하는 것이 중요한데, early stopping 을 설정해놓으면 그럴 필요가 없다.

​	이렇게만 보면 early stopping을 적용하지 않을 이유가 없어보인다. 하지만 early stopping을 그다지 추천하지 않는다는 의견도 존재하는데, 실제로 validation error가 그다지 변화하지 않는 상태 - model has converged - 에서 더 훈련을 진행했을 때 좋은 결과가 더러 나올 뿐만 아니라 어떻게, 얼마나 전체 과정에 영향을 줄지 알지 못하기 때문이다.



## 2. Weight Decay

​	overfitting을 막기 위해선 모델이 지나치게 complex해지는 것을 방지하는게 필요하다. 따라서 complexity에 따라 모델에 penalty를 줄 수 있는데, 일반적으로 loss 함수에 모든 param값(weights)을 더해(정확히는 squared ver) parameter가 지나치게 많아지는 것을 막는다. parameter가 많아질 수록 모델은 점점 complex 해진다. 

그러나 squared norm of parameters를 loss에 추가하게 되면 loss가 지나치게 커지는 문제가 생기는데, 이때문에 어떤 작은 숫자를 곱할 수 있다. 그리고 이 곱해진 숫자를 **weight decay** 라고 한다. 전체 loss 식은 아래와 같다.

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