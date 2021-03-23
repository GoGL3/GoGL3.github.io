---
title: "[DL 101] Learning Rate Scheduler"
excerpt: "What is learning rate scheduler?"
date: 2021-02-25 16:000 -0400
author : 오승미
use_math: true
categories :
  - deep-learning
tags :
  - learning-rate-scheduler
  - deep-learning

---

#   Learning Rate Scheduler

We can adjust the learning rate depending on *some conditions* which helps to improve the model performance. We are going to briefly check some popular methods in this post.

## 1. Lambda LR

$ lr_{epoch} = lr_0*\lambda $

where lr_lambda is a **function** or list(of functions to each group of parameters) and it is multiplied by the initial learning rate.

Pytorch:

```python
lambda1 = lambda epoch: epoch // 30
lambda2 = lambda epoch: 0.95 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
```



## 2. Step LR

$$ lr_{epoch} =
\begin{cases}
  \gamma*lr_{epoch-1} & \text{if epoch * step_size = 0} \\
  lr_{epoch-1} & \text{otherwise}
\end{cases} $$

​	Decays the learning rate of each parameter by gamma every step_size epochs. Normally, set the step_size to be five epochs; meaning that decaying the lr every five epoch. Assume that the initial learning rate is 0.05 for all groups, step_size = 2 and gamma = 0.1, lr = 0.005 if 2 <= epoch < 4, lr = 0.0005 if 4 <= epoch < 6, so on.

​	Pytorch:

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
```

​	Also, add "scheduler.step()" after "optimizer.step()" in the train step.

```python
outputs = model(X)
loss = criterion(outputs, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()
```

​

## 3. Exponential LR

$ lr_{epoch} = \gamma * lr_{epoch-1} $

Decays the learning rate by gamma every epoch.

Pytorch:

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
```



## 4. CosineAnnealingLR

$ \eta_{t}=\eta_{\min }+\frac{1}{2}\left(\eta_{\max }-\eta_{\min }\right)\left(1+\cos \left(\frac{T_{c u r}}{T_{\max }} \pi\right)\right) $

​	Decays the learning rate using a cosine annealing schedule; thus combining both periods with hot learning rates and cold learning rates - improving model performance.





## Reference

- https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
- https://towardsdatascience.com/learning-rate-scheduler-d8a55747dd90
- https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
