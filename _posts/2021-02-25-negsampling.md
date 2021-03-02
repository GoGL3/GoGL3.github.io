---
title: "[RecBasics] Negative samples in Top-K recommender task (Part 1)"
date: 2021-02-25 10:00 -0400
author : 정여진
categories :
  - recommendation-system
tags :
  - recommendation-system
  - negative-sampling
  - deep-learning

---


In this post, we will dive into negative sampling for test / train instances in recommendation system.

## Explicit and Implicit Feedback

Explicit feedback is interactions where users express definite preferences of each item i.e. ratings of scale 1-5. Implicit feedback is, on the other hand, interactions through which we can infer users' preferences. For example total views, downloads, site visits can be implicit feedback. We can infer that the user showed at least some interest in a particular item.

## Why negative samples?

In real life, explicit feedback is not always available. In most cases, we have to recommend items based on users' implicit feedback. One important issue of implicit feedback is that **only positive interaction is available**. For explicit feedback, such as ratings from 1-5, we have complete data of both items _liked_ and _not liked_ by users. However for implicit feedback we only have items _interacted_ by users, in other words, we only have positive interactions. This one-level data cannot be learned by any algorithms because all data is labelled 1 (think of binary classification task). This is why we have to create unobserved data with labels 0. These unobserved interactions comprise of both real negative feedback (user saw the item but did not click) and missing values (it just did not pop up to user's screen).

## Negative sampling mechanism for learning implicit feedback

Assume that _U_ is space of all possible interaction pairs. For example, if there are 10 users and 10 items in data, there will be 100 _(u,i)'s in U_. Also suppose _U*_ is observed(positive) interaction pairs. Then _(u,i) in U\U*_ will be unobserved pairs.

In the most simple scheme, we just assume all interaction pairs in _U\U*_ are real negative samples. We preset a _num_neg = k_ number and for each _(u,i) in U*_, we uniformly sample _k_ negative interactions and label them 0.

```python
def get_train_instances(uids, iids, num_neg, num_items):
        """
        :objective: create negative samples in training set
        """
        user_input, item_input, labels = [],[],[]
        zipped = set(zip(uids, iids)) # train (user, item) 세트

        for (u, i) in zip(uids, iids):

            # positive interaction
            user_input.append(u)  
            item_input.append(i)  
            labels.append(1)     # label is 1

            # negative interaction
            for t in range(num_neg):

                j = np.random.randint(num_items)      
                while (u, j) in zipped:           # uniformly sample (u,j) not in data
                    j = np.random.randint(num_items)  

                user_input.append(u)  
                item_input.append(j)  
                labels.append(0)      # label is 0

        return user_input, item_input, labels
```


If more auxiliary information is available, we can sample more reliable negative instances such as _clicked but not bought_ items. Other methods include oversampling hard negative samples (dynamic negative sampling, DNS) and frequency-based sampling. In 2019, (Ding, 2019) proposed a reinforcement learning based negative sampling (RNS) that generates exposure-alike negative instances. 


_In Part2, we will look at how test data of Top-k recommender task is generated._


## References
- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2012). BPR: Bayesian personalized ranking from implicit feedback. arXiv preprint arXiv:1205.2618.
- Ding, J., Quan, Y., He, X., Li, Y., & Jin, D. (2019, August). Reinforced Negative Sampling for Recommendation with Exposure Data. In IJCAI (pp. 2230-2236).



---
