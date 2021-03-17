# 1. TX2上命令
TX2板载相机型号OV5693 (2592*1944，比例4: 3)

> 参见： https://blog.csdn.net/qq583083658/article/details/86543316  
* ubuntu 16.04开风扇`sudo  ~/jetson_clocks.sh`	; 18.04直接`sudo jetson_clocks`  
* 关风扇需要先store未开时的状态，之后restore则可以关风扇。  
store命令`sudo  ~/jetson_clocks.sh --store`，restore命令`sudo  ~/jetson_clocks.sh --restore`	  
* 查询工作模式`sudo nvpmodel -q verbose`  
* 修改为性能模式`sudo nvpmodel -m 0`  
* 查CPU占用率`top`, GPU占用`./tegrastats`， TX2上用不了nvidia-smi
* 查看L4T版本`head -n 1 /etc/nv_tegra_release`  
* 查看系统内核`uname -a`
* 查看资源占用和版本：`sudo pip install jetson-stats`，安装之后可以使用三个命令：`jtop`, `jetson_release`, `jetson_variables` (Jetpack 4.2上可行)

# 2. TX2上软件安装
* OpenCV/Pytorch/EdgeX： 因为TX2是armv8构架，对于L4T R28系统版本(Ubuntu16.04)来说，其不支持从pip安装OpenCV`pip3 install opencv-python --user`，只能从源码编译，参见： https://github.com/Hydroxy-OH/NVIDIA-Jetson-TX2  
* ROS Kinetic的安装： https://github.com/sxontheway/Keep-Learning/blob/master/ROS/ros_basic.md  
* Jetpack3.3，Scipy通过pip安装奇慢无比(20-30分钟)。但用`sudo apt-get install python3-scipy`安装，版本会过低。
* 刷了Jetpack4.2之后，安装十分方便。  
刷机见：https://developer.nvidia.com/embedded/jetpack  
pytorch 安装见:https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/  
pip,numpy,tensorflow,deepstream,tensorrt在刷机时就可以安好  
torchvision安装：sudo pip3 install torchvision  
scikit-learn安装: 需要先安装`sudo apt-get install gfortran`, `sudo pip3 install Cython`
VSCode安装及Romote Control：https://www.jetsonhacks.com/2019/10/01/jetson-nano-visual-studio-code-python/ 

# 3. TX2上两个cores被禁用
> https://forums.developer.nvidia.com/t/cannot-enable-denver-cores-for-tx2-jetpack-4-4-dp/124708/48

Jetpack 4.4, Jetpack 4.5 用 `jtop`可能出现两个核denver cores被禁用的情况，解决方案：
* 将 `/boot/extlinux/extlinux.conf` 中的 `isolcpus=1-2` 改为 `isolcpus=`
* reboot
* 查看 `cat /proc/cmdline`，显示`isolcpus=`，而不是 `isolcpus=1-2`；查看`jtop`，所有核都被启动

实际上，被禁用的两个核是Denver内核，因为TX2的调度机制使得两个Denver的内核优先级比四个Cortex-A57优先级高，但是他们其实更慢，所以禁用这两个cores其实对大多数任务（在四个A57核足够应对时）有性能提升，并且可以降低功耗。所以可以先执行上述修改启用这两个核，再通过 `jtop` 中 disable jetson_clocks 关闭两个 Denver 核（需要时可以随时打开）

