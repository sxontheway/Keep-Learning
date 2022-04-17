# Lidar
## RGB + point cloud Fusion
* MV3D, AVOD：先投影到某个视图上，再融合：投影会损失信息
* PointFusion：RGB 和 point cloud 各自分别经过 Resnet 和 PointNet，再融合
## HDNet: pointcloud + HD map
* 用 U-net 从 BEV 生成 Semantic Map（道路区域的mask）
* 查询点云中每一个点的高度以及该点投影在地面的高度，两者作差，以抵消路面倾斜这种情况
* 将（Semantic Map）和（抵消路面高度后的BEV在z方向的切片）一起作为网络输入

## PointNet, PointNet++, F-PointNet, PointRCNN 思想
* PointNet
  * 输入为整个点云    
    整个分类/分割任务建模为：`f({x1, x2, ..., xn}) ≈ g(h(x1), ..., h(xn))`; g用maxpooling实现，h用MLP实现
  * 将点云映射到高维空间：MLP
  * 旋转、平移不变性：用 T-net 来学旋转矩阵（T-net之中也用了maxpooling来保证对输入点云顺序的鲁棒）
  * 点云的输入顺序鲁棒：maxpooling
  * 同时利用局部和全局信息：concat point-wise feature and global feature  
  （将 classification network 的结果 concat 到 segmentation network (逐点判断类别)）
  
* PointNet++：多尺度，多分辨率，更好的局部特征提取
  * 有点像U-net的思想：https://zhuanlan.zhihu.com/p/44809266
  
* F-PointNet: https://zhuanlan.zhihu.com/p/41634956
  * 2D image object detection 生成 bbox，得到3D的Frustum 
  * 对 3D Frustum 内的 point 进行 segmentation
  * 对 a subset of segmented point cloud 进行 3D bbox regression：Pointnet(Feature Extractor) + FC(Head)  
  （类比 Faster-RCNN 的 bbox regression）
    <p align="center" >
    <img src="./pictures/fpointnet.jpg" width="800">
    </p>
    
* PointRCNN: https://zhuanlan.zhihu.com/p/84335316
  * Stage 1
    * 用了 PointNet++ 先逐点进行 segmentation（foreground or background)：Focal Loss
    * Bin-based 3D Box Generation: Anchor-free  
    对于每一个 foreground point，生成一个 3d box proposal；  
    利用了PointNet++提取的特征，loss涉及object中心所在的bin
    * 用 NMS 剔除多的Proposal，进入bbox refinement阶段
  * Stage 2
    * 先在xyz轴上扩大每个Proposal（框住更大的体积）
    * concat之前的 Semantic feature 和新得到的 local feature 
    * PointNet++ 进行 regression
    <p align="center" >
    <img src="./pictures/pointrcnn.jpg" width="800">
    </p>
# Radar 
## Mapping
* See Through Smoke: Robust Indoor Mapping with Low-cost mmWave Radar_Mobisys20
* Material-based Segmentation Mapping and Refinement via Millimeter-Wave Radar_Mobicom20
* SuperRF: Enhanced 3D RF Representation Using StationaryLow-Cost mmWave Radar_EWSN20
* High Resolution Millimeter Wave Imaging For Self-Driving Cars
## Detection
* Semantic Segmentation on Radar Point Clouds_Fusion18
* 2D Car Detection in Radar Data with PointNets_ITSC19
* Deep Learning Based 3D Object Detection for Automotive Radar and Camera_EuroRadarConf19
* Automotive Radar Dataset for Deep Learning Based 3D Object Detection_EuroRadarConf19
* MRsense: Accurate 3D Bounding Box Estimation with Multi-Radars
