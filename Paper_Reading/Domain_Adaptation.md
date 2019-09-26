# Domain Adaptive Object Detection
Let's divide the methods in domain adaptation using number of networks:

|Number of Networks Used in Total|Number of Networks to be Trained|Description|
| :------------: |:---------------:|:-----:|
| 1 | 1 | Self-Learning, Pseudo-Labeling, Regularized Tranfer Learning|
| 2 | 1 | Knowledge Distillation, Feature Learning |
| 2 | 2 | Co-Teaching |

<br>

* Few-shot domain adaptation  
    * `LSTD: A Low-Shot Transfer Detector for Object Detection_AAAI18` : Faster-RCNN + SSD + Extra Modules
    * `Few-shot Adaptive Faster_RCNN_CVPR19` :  Extra Modules

* Zero-shot domain adaptation
    * Feature Learning Based 
        * `Domain Adaptive Faster R-CNN for Object Detection in the Wild_CVPR18` : Add extra modeles
    * Pseudo-Labeling Based 
        * `Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation_CVPR18` : GAN-based domain transfer + pseudo labeling  
        * `Automatic adaptation of object detectors to new domains using self-training_CVPR19` : Pseudo-Labeling + Score mapping, detect only one class 
        * `A Robust Learning Approach to Domain Adaptive Object Detection_ICCV19` : How to train with noisy labels (robust learning)
        * `Training Object Detectors With Noisy Data_IV19` : co-teaching  

* Other Related Topics
    * Training on the Edge
        * `Training on the Edge: The why and the how_IPDPSW19`
        * `Performance Analysis and Characterization of Training Deep Learning Models on Mobile Devices_ArXiv19` : Deep learning model training on TX2
    * Incremental Learning on the Edge   
        * `RILOD: Near Real-Time Incremental Learning for Object Detection at the Edge_SEC19` : Pseudo labeling + Incremental learning
        * `In-situ AI: Towards Autonomous and Incremental Deep Learning for IoT Systems_HPCA18`
    * Automatically Labeling
        * `Mark Yourself: Road Marking Segmentation via Weakly-Supervised Annotations from Multimodal Data_ICRA18`
