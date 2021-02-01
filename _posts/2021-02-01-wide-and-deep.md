---
title: "[Paper Review] Wide & Deep Learning for Recommender Systems (2016)"
date: 2021-01-31 14:000 -0400
author : 정여진
categories :
  - recommendation
  - deep-learning
---

Today, I will review a paper that first integrated the idea of latent variables with deep learning. The original paper was written by Heng-Tze Cheng et al. from Google Inc ([Google](https://arxiv.org/abs/1606.07792)). First introduced in 2016, this method was successfully implemented in Google Play app store (~~though it now seems to have been replaced or enhanced by DeepMind's sota DL methods. we will go over that later~~ ). You can also find open-source implementation in tensorflow. Before going deeply into this paper, let's go over how Wide & Deep method came to be.

## Background Information
Keeping track of frequent co-occurring feature combinations is important in quality of recommendation. (By features, I mean user, item, and contextual information.) This co-occurrence is termed as _cross-product_ features. Cross-product terms can incorporate non-linear interactions between features. For example, system memorizes interaction of _color_ and _brand_ when recommending _sneakers_. However, (generalized) linear models fail to include interaction terms as they have to be manually included by the author. Therefore, hidden interactions are often disregarded. To overcome such limitations, **Factorization Machines(FM)** arose as an effective method to model up to degree-2 interactions as inner product of latent vectors, and showed very promising results.


Another important part in recommendation system is whether unseen, new user-item interactions can be modelled. FM can generalize to unseen interactions by learning low-dimensional latent vector representation of users and items. **Deep learning** recommendation methods emerged to improve representation of features through low-dimensional embedding layers. Another advantage of deep learning method is that high-order cross-product terms can be added to include sophisticated relationships between features.

A hybrid approach, **Wide & Deep method** succeeds in memorizing wide cross-product features, and generalize unseen feature interactions through low dimensional embeddings.



## Paper Review
Two important concepts appear in this paper : _Memorization and generalization_

### Memorization - Wide - past interactions

  *"Memorization can be loosely defined as learning the frequent co-occurrence of items or features and exploiting the correlation available in the historical data."*



### Generalization - Deep - unseen interactions

  *"Generalization, on the other hand, is based on transitivity of correlation and explores new feature combinations that have never or rarely occurred in the past."*

One problem of cross-product modelling is that it fails to consider feature pairs that have not appeared in training data. Systems based on memorization soley would make the suggestions limited and boring. Users might want to find new delights from new items. **Deep** part effectively learns low embedding representations of features to generalize user-item interactions.


### Joint learning
Key advantage of this method is that wide part and deep part are jointly learned, rather than simple ensemble.




### Model Structure


## Further Improvements

There were massive studies on applications of deep-learning-based recommendation systems after this paper was proposed. Some examples are _Production-based-Neural-Network(PNN,2016)_ and _Factorization_Machine-supported-Neural_Network(FNN,2017)_. However, these fail to capture low-order interactions that Wide & Deep method could. _DeepFM(2017)_ was created to overcome limitation of feature-engineering in Wide & Deep method.
