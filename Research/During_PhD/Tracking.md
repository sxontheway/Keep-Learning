# SOT (Single Object Tracking)
KCF/ SiameseFC, CFNet, DCFNet/ MDNet/ GOTURN

# MOT (Multiple Object Tracking)
> Input: detection responses; Output: tracklets or even trajectories
## Overview
* Multiple Object Tracking: A Literature Review_ArXiv14  
* The 2018 NVIDIA AI City Challenge_CVPRW18  
* CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification_CVPR19  

## Tracking by detection 
### Detection: 
Yolo, SSD, Faster-RCNN, HOG/DPM(Deformable Part Model)  
### Offline(Batch) Association
> Two common frameworks:
* (1) Hierarchical association: detection -> tracklets -> trajectories
    * `Detections to tracklets`: Closed-Loop Tracking-by-Detection for ROV-Based Multiple Fish Tracking_16
    * `Tracklets to trajectories`: Single-camera and inter-camera vehicle tracking and 3D speed estimation based on fusion of visual and semantic features_CVPRW18
    * `The whole pipeline`: Robust Object Tracking by Hierarchical Association of Detection Responses_ECCV08
* (2) Global optimization in a graph-based representation, eg. a network flows  
    * Global Data Association for Multi-Object Tracking Using Network Flows_CVPR08  
    * `Siamese CNN + Gradient Boosting + Network Flow: Learning by tracking`: Siamese CNN for robust target association_CVPRW16  
    * `Use DNN to calculate edges' costs`: Deep Network Flow for Multi-Object Tracking_CVPR17

### Online Association
> Strictly speaking, online is a special case of offline, so the following methods can also be applied in offline tracking.
* Probabilistic: MCMC/ Particle Filter
* Deterministic:
   * Hungarian Algorithm/ KM Algorithm/ Bipartite Matching: 
      * `Hungarian + Kalman filter`: SORT/DeepSORT  
      * `Hungarian + Tracklet confidence`: Robust Online Multi-object Tracking Based on Tracklet Confidence and Online Discriminative Appearance Learning_CVPR14
   * Markov Decision Process(RL-based): 
      * learning to Track: Online Multi-Object Tracking by Decision Making_ICCV15
* Others:  
   * `LSTM`: Online Multi-Target Tracking Using Recurrent Neural Networks_AAAI17  
   * `Background deduction + KCF`: Multiple Object Tracking with Kernelized  Correlation Filters in Urban Mixed Traffic_CRV17  
   * `Optimal Bayes Filters`: Towards a Principled Integration of Multi-Camera  Re-Identification and Tracking through Optimal Bayes Filters_CVPRW17  

### Motion Prediction:  
* `Decay model`: CaTDet: Cascaded Tracked Detector for Efficient Object Detection from Video  
* `Kalman Filter`: SORT/DeepSORT

## Evaluation Metrics
[MTMC Tracking 评价指标](https://zhuanlan.zhihu.com/p/35391826)

# Resources
[Deep learning for tracking and detection](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection)  
[Multi object tracking paper list](https://github.com/SpyderXu/multi-object-tracking-paper-list)  
[SSD+SORT](https://github.com/SpyderXu/ssd_sort)  
[Yolo+DeepSORT](https://github.com/Qidian213/deep_sort_yolov3)  
[IOU Tracker](https://github.com/bochinski/iou-tracker)  
[KIOU Tracker](https://github.com/siyuanc2/kiout)
