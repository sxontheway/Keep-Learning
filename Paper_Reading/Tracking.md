# SOT (Single Object Tracking)
KCF/ SiameseFC, CFNet, DCFNet/ MDNet/ GOTURN

# MOT (Multiple Object Tracking)
## Overview
* Multiple Object Tracking: A Literature Review_ArXiv14  
* The 2018 NVIDIA AI City Challenge_CVPRW18  
* CityFlow: A City-Scale Benchmark for Multi-Target  Multi-Camera Vehicle Tracking and Re-Identification_CVPR19  

## Detection + Association paradigm 
Detector: Yolo, SSD, Faster-RCNN, HOG/DPM(Deformable Part Model)  

Associator:  
* Probabilistic:  
    * Kalman Filter + Hungarian Algorithm: SORT/DeepSORT  
    * Decay model: CaTDet: Cascaded Tracked Detector for Efficient Object Detection from Video  
    * Simple tracklet generation: Closed-Loop Tracking-by-Detection for ROV-Based Multiple Fish Tracking

* Deterministic:  
    * Global Data Association for Multi-Object Tracking Using Network Flows_CVPR08  
    * MDP tracking: Learning to Track: Online Multi-Object Tracking by Decision Making_ICCV15
    * Siamese CNN + Gradient Boosting: Learning by tracking: Siamese CNN for robust target association_CVPRW16  
    * Use DNN to generate edges' costs: Deep Network Flow for Multi-Object Tracking_CVPR17
    * Tracklet Association: Single-camera and inter-camera vehicle tracking and 3D speed estimation based on fusion of visual and semantic features_CVPRW18


## Other Works
* Background deduction + KCF: Multiple Object Tracking with Kernelized Correlation Filters in Urban Mixed Traffic_CRV17  
* LSTM: Online Multi-Target Tracking Using Recurrent Neural Networks_AAAI17  
* Optimal Bayes Filters: Towards a Principled Integration of Multi-Camera Re-Identification and Tracking through Optimal Bayes Filters_CVPRW17  


## Evaluation Metrics
[MTMC Tracking 评价指标](https://zhuanlan.zhihu.com/p/35391826)

# Resources
[Deep learning for tracking and detection](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection)  
[Multi object tracking paper list](https://github.com/SpyderXu/multi-object-tracking-paper-list)  
[SSD+SORT](https://github.com/SpyderXu/ssd_sort)