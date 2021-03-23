---
title: "[DL 101] Transfer Learning  vs.  Fine Tuning  vs.  Training from scratch"
excerpt: "What is transfer learning and why?"
date: 2021-02-12 15:000 -0400
author : 오승미
categories :
  - deep-learning

tags :
  - deep-learning
  - transfer-learning
  - fine-tuning
  - Training from scratch
  - CNN


---

# Transfer Learning  vs.  Fine Tuning  vs.  Training from scratch

This article is based on [CS231n note](https://cs231n.github.io/transfer-learning/) and [keras guides](https://keras.io/guides/transfer_learning/).



## 1. Training from scratch

​	Build a CNN model by stacking layers from beginning to end.

## 2. Transfer Learning

​	Actually, however, it is not easy to have enough datasets for training. Therefore, in many cases, convolutional networks are often built on models that are already sufficiently trained with large data. Not only does it save time, but it can also improve performance even more.

**Steps**

 1. Choose a pretrained model. (resnet, alexnet, googlenet etc)

 2. *Freeze* all layers of the base model. We should set all the information(weights) in the pretrained model not to be updated via training.

    In pytorch, you can implement this through defining a simple function,

    ```python
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    ```

 3. Create a model to train and add it to 2.

 4. Train added part (that we created in step3) in the full model.

    In pytorch, I built a function like below, using squeezenet,

    ```python
    def initialize_model(num_classes, feature_extract, use_pretrained = True):
        model_ft = models.squeezenet1_0(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size = 1, stride = 1)
        model_ft.num_classes = num_classes

        return model_ft
    ```

## 3. Fine Tuning

​	If we can tell that the model has converged to the new data well after the transfer learning, we can re-train the whole model, including the pretrained part, by unfreezing the layers from the pretrained model. Although performance can be tremendously improved, the problem of overfitting exists, so it is only necessary to apply it with very low learning rates.



## Then, when and how to use those concepts properly?

​	There are four scenarios based on the *size* of the new dataset and the *similarity* of the original dataset. Let's look at how each of the three methods should be applied.

​	**1. New dataset is small and similar to the original**  Rather than fine-tuning the ConvNet, it is appropriate to use the last extracted features of the convolutional layer and train only the linear classifier  because of overfitting problem.

​	**2. New dataset is large and similar to the original** Since the data is enough, we can apply fine-tune.

​	**3. New dataset is small but very different from the original**  Appropriate to train the linear classification only.

​	**4. New dataset is large and very different from the original** If you have enough data size, you can train from scratch. However, the weights of the pretrained model still can be applied during the initialization phase.
