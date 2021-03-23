---
title: "Pose Estimation - AlphaPose and its application"
excerpt: "[AlphaPose](http://www.mvig.org/research/alphapose.html) is an accurate multi-person pose estimator, which is the first open-source system that achieves 70+ mAP (75 mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset."
date: 2021-03-22 06:000 -0400
author : 오승미
use_math: true
tags :
- computer vision
- pose estimation
- object detection


categories:
- computer vision
- deep-learning

---

[AlphaPose](http://www.mvig.org/research/alphapose.html) is an accurate multi-person pose estimator, which is the first open-source system that achieves 70+ mAP (75 mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset.

[RMPE](https://arxiv.org/abs/1612.00137) is a popular top-down method of Pose Estimation. The authors posit that top-down methods are usually dependent on the accuracy of the person detector, as pose estimation is performed on the region where the person is located. Hence, errors in localization and duplicate bounding box predictions can cause the pose extraction algorithm to perform sub-optimally.

To resolve this issue, the authors proposed the usage of Symmetric Spatial Transformer Network (SSTN) to extract a high-quality single person region from an inaccurate bounding box.

A Single Person Pose Estimator (SPPE) is used in this extracted region to estimate the human pose skeleton for that person.

A Spatial De-Transformer Network (SDTN) is used to remap the estimated human pose back to the original image coordinate system.

Finally, a parametric pose Non-Maximum Suppression (NMS) technique is used to handle the issue of redundant pose deductions.

Furthermore, the authors introduce a Pose Guided Proposals Generator to augment training samples that can better help train the SPPE and SSTN networks. The salient feature of RMPE is that this technique can be extended to any combination of a person detection algorithm and an SPPE.

If you wanna try, read their [GitHub](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md) - well organized and clear to follow!



## Applications  

So, how did I



## References

- https://medium.com/beyondminds/an-overview-of-human-pose-estimation-with-deep-learning-d49eb656739b
- https://github.com/MVIG-SJTU/AlphaPose
-
