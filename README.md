# ADG-Net: Adaptive Detail Guidance Network for Breast Cancer Ultrasound Image Segmentation
### Abstract
Cancer is always a major public health problem that threatens human health, especially in most countries around the world, breast cancer continues to top the list of female tumors with a high mortality rate. Although ultrasound imaging plays an important role in early breast cancer detection, the characteristics of breast cancer, such as its blurred boundaries and irregular shape, make achieving accurate breast tumor segmentation a challenging task. To address these challenges, we innovatively design the Adaptive Detail Guidance Network (ADG-Net) for breast cancer image segmentation. Specifically, the following three modules are included: Dual-Axis Self-Calibrating Attention (DASCA), Adaptive Information Fusion Module (AIFM), and Multiscale Edge-Enhanced Gradient Alignment Module (MEEGA). Concretely, DASCA enhances the spatial feature reconstruction capability of the network and enriches the local detail information; AIFM takes a random selection strategy to help the network dig deeper and learn more complex potential features, which significantly improves the generalization ability of the network; MEEGA utilizes the Laplace function to target edge-related features for anti-noise processing, which effectively solves the weak boundary problem and further enriches the segmentation prediction results of decoders at all levels. To validate the effectivity of the network, we conduct comprehensive experiments on three authoritative publicly available medical image datasets. Experimental results show that our approach presents significant performance advantages over current advanced baseline methods.
### Overall
![](https://github.com/jiazhuangdiandian/ADG-Net/blob/master/img/1.jpg?raw=true)
### Segmentation results
![](https://github.com/jiazhuangdiandian/ADG-Net/blob/master/img/2.jpg?raw=true)
### Getting Started
**Environment**

1.Clone this repo:https://github.com/jiazhuangdiandian/ADG-Net.git

2.Create a new conda environment and install dependencies.

pip:

    - addict==2.4.0
    - dataclasses==0.8
    - mmcv-full==1.2.7
    - numpy==1.19.5
    - opencv-python==4.5.1.48
    - perceptual==0.1
    - pillow==8.4.0
    - scikit-image==0.17.2
    - scipy==1.5.4
    - tifffile==2020.9.3
    - timm==0.3.2
    - torch==1.7.1
    - torchvision==0.8.2
    - typing-extensions==4.0.0
    - yapf==0.31.0
  
**Training & Test**

Python train.py

Python val.py

**Datasets**

Breast Ultrasound Dataset B: https://helward.mmu.ac.uk/STAFF/M.Yap/dataset.php

Breast Ultrasound Images (BUSI)：https://github.com/hugofigueiras/Breast-Cancer-Imaging-Datasets

BLUI:https://qamebi.com/breast-ultrasound-images-database/
