# 1. TX2上命令
TX2板载相机型号OV5693 (2592*1944，比例4: 3)

> 参见： https://blog.csdn.net/qq583083658/article/details/86543316  
* 开风扇`sudo  ~/jetson_clocks.sh`	  
* 关风扇需要先store未开时的状态，之后restore则可以关风扇。  
store命令`sudo  ~/jetson_clocks.sh --store`，restore命令`sudo  ~/jetson_clocks.sh --restore`	  
* 查询工作模式`sudo nvpmodel -q verbose`  
* 修改为性能模式`sudo nvpmodel -m 0`  
* 查CPU占用率`top`, GPU占用`sudo ～/tegrastats`。 TX2上用不了nvidia-smi

 
# 2. TX2上软件安装
* OpenCV/Pytorch/EdgeX： 因为TX2是armv8构架，对于L4T R28系统版本(Ubuntu16.04)来说，其不支持从pip安装OpenCV`pip3 install opencv-python --user`，只能从源码编译，参见： https://github.com/Hydroxy-OH/NVIDIA-Jetson-TX2  
* ROS Kinetic的安装： https://github.com/sxontheway/Keep-Learning/blob/master/ROS/ros_basic.md  
* Jetpack3.3，Scipy通过pip安装奇慢无比(20-30分钟)。但用`sudo apt-get install python3-scipy`安装，版本会过低。
