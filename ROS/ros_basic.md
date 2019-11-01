# 1. ROS基础
## 1.1 安装
对于TX2这种Arm64平台， 如果系统是Ubuntu16.04， ROS kinetic只能通过源码安装， 参见：https://github.com/jetsonhacks/installROSTX2  
对于Amd64平台， 或Debian系统的Arm64平台， ROS kinetic/melodic都可通过apt-get直接安装，参见：http://wiki.ros.org/melodic/Installation/Debian

---
<br>

## 1.2 Python3 in ROS
### 1.2.1 ROS下cv2版本冲突的问题
* 问题起源： 详见： [After install ROS Kinetic, cannot import OpenCV?](https://stackoverflow.com/questions/43019951/after-install-ros-kinetic-cannot-import-opencv)  
  * 用`pip3 install opencv-python --user`，安装了基于python3的OpenCV后，在python3下`import cv2`没问题  
  * 之后又安了ROS Kinetic， 再在python3下`import cv2`报错：`/opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type`
* 原因：  
  * ROS Kinetic内的OpenCV是基于python2的。 在没有安装ROS时，系统中只有Python3的OpenCV版本，不会出错； 但是在装了ROS之后，ROS的OpenCV环境变量优先级比python3的OpenCV优先级高。
  * 查看环境变量优先级命令， 发现ROS的路劲优先级更高
    ```python
    import sys, pprint;  
    pprint.pprint(sys.path)
    ```  
* 解决方案二一（最实用）：  
在import cv2时稍微绕一下：
  ```python
  import sys
  ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
  if ros_path in sys.path:
      sys.path.remove(ros_path)
  import cv2
  sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
  ``` 
* 解决方案二：   
重命名ros自带的cv2.so， 但要注意以后import ROS自带的cv2时， 要执行`import cv2_ros`， 详见： [cv_bridge](https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3/50291787#50291787)
  ```bash
  cd /opt/ros/kinetic/lib/python2.7/dist-packages/
  sudo mv cv2.so cv2_ros.so
  ```
* 解决方案三：  
注释掉`~/.bashrc`中的`source /opt/ros/kinetic/setup.bash`， 但这样导致ROS和Python3不能同时使用
* 其他方案，例如使用软链接：  
  详见： [After install ROS Kinetic, cannot import OpenCV?](https://stackoverflow.com/questions/43019951/after-install-ros-kinetic-cannot-import-opencv)  

### 1.2.2 在Python3下使用cv_bridge
> 参见 [cv_bridge： OpenCV和ROS之间的桥梁](./cv_bridge)
* ROS Kinect自带的cv_bridge只支持python2，但tensorflow/pytorch要用python3，这会导致一些问题。
例如在一个.py脚本中，没办法同时使用cv_bridge和pytorch，也即没法对image话题的rosmsg进行Detection之类的工作。   
解决方案是：下载cv_bridge的源码、将其作为一个包放入工作空间、指定cmake用python3编译它。详见`cvbridge.md`文件。
* 用cv_bridge比较耗时: 如果图像捕获节和识别是由一个节点完成，用VideoCapture可以省去图像发布、订阅、cv_bridge转换三步。
但当有多个节点需要image信息，或想要一个节点专门负责图像的捕获和发布时，cv_bridge是不能替代的，参见：  
  [记录一下TX2，ros，Python，opencv与TensorFlow的血海深坑](https://blog.csdn.net/weixin_42048023/article/details/89502641)  
  [CSI Cameras on the TX2 (The Easy Way)](http://petermoran.org/csi-cameras-on-tx2/)

### 1.2.3 Python3下ROS其他包的安装
上面一节是安了两个版本的OpenCV， 这一节讲在基于python2的ROS包没有安装时， 怎么直接安python3的版本  
* 主要步骤：
  * 安装虚拟环境并启动
  * 在虚拟环境中安装所需的包， 例如：rospkg， catkin-tools，
    ```bash
    pip3 install catkin-tools
    pip3 install rospkg
    ```
  * 在节点的python文件开头加`#!/usr/bin/env python3`
  * 启动python3的venv， 再rosrun  

* 需要注意的是，venv只改变python环境变量（也即包的查找路劲）。 与用什么解释器，是python2还是python3无关（除非在建立venv时就用了-p命令）。
即使使用python3的venv， 但是只要不在`.py`文件中添加python3的shebang， rosrun还是会调用python2的， 这样可以很方便地兼容原来Python2写的代码。  
* `#!/usr/bin/env python3`指定了解释器为python3，只对可执行文件适用。  在bash中显式指定用python3：`python3 a.py`，这时不管a.py的权限是不是可执行的，a.py都会被用python3执行，即使在文件开头加了`#!/usr/bin/env python2`，也不会有用了。
> 参见： [在ROS中使用Python3（1）](https://community.bwbot.org/topic/499/%E5%9C%A8ros%E4%B8%AD%E4%BD%BF%E7%94%A8python3),  [在ROS中使用Python3（2）](https://www.cnblogs.com/h46incon/p/6207145.html)

### 1.2.4 rospy脚本的运行
* 脚本应该放在`scripts`文件夹中，并为该脚本设置`chmod +x`权限
* .launch文件应该放在launch文件夹中，如下三个参数分别为包名、节点名、脚本文件名
  ```xml
  <launch>
      <node pkg = "micro_doppler_pkg" name = "micro_doppler_pkg" type = "micro_doppler.py" output="screen" />
  </launch>
  ```
* 在Kinetic版本中，因为rospy基于python2编译，如果脚本指定编译器python3，`import rospy`会报错  
  解决方案`sudo apt-get install python3-catkin-pkg-modules`，`sudo apt-get install python3-rospkg-modules`

### 1.2.5 catkin相关
* `catkin config`查看空间配置信息
* `catkin build + <package_name>`可以选择对单个包进行build。`catkin build`和`catkin_make`都会对src中的所有包进行build。 二者在同一工作空间内不能混用
* `catkin clean`可以清除编译生成的build，devel，logs等文件夹
* `.catkin_tools`这个隐藏文件记录了工作空间的配置信息。如果不删除它，会在`catkin init`时出现`catkin workspace ... is already initialized. No action taken.`
---
<br>

## 1.3 ROS的命名空间
ROS下有4类命名空间： base, global, relatove, private

| 类型 | 格式 | 
| :----- |:----:| 
| base | <node_name> |
| global | /<node_name> |
| relative | /.../<node_name> |
| private | ~<node_name> |


* 节点的命名空间  
  * 节点的命名空间会影响其发布的话题： 参见： https://blog.csdn.net/u014587147/article/details/75647002
    ```bash
    # turtlesim是package name, turtlesim_node是执行的文件名
    rosrun turtlesim turtlesim_node
    # 文件中 ros::init(argc, argv, "turtlesim"); 这句建立了节点 /turtlesim

    rosrun turtlesim turtlesim_node __ns:=/my
    # 指定了命名空间为/my，故建立的节点名为 /my/turtlesim

    rosrun turtlesim turtle_teleop_key
    # 并不能控制小乌龟， 因为turtle_teleop_key只会发布/turtel1/cmd_vel话题， 不会发布/my/turtel1/cmd_vel

    rosrun turtlesim turtle_teleop_key /turtle/cmd_vel:=/my/turtle/cmd_vel
    # 进行话题重映射，小乌龟受控了
    ```

* 句柄的命名空间
  ```bash
  ros::NodeHandle nh;           # nh的命名空间为 /<node_namespace>
  ros::NodeHandle nh("alex");   # nh命名空间为 /<node_namespace>/alex
  ros::NodeHandle nh("~");      # nh命名空间为 /<node_namespace>/<node_name>
  ros::NodeHandle nh("~alex");  # nh命名空间为 /<node_namespace>/<node_name>/alex
  ```
  * `节点名(<node_name>)`和`节点的命名空间(<node_namespace>）`是不同的
  * <node_namespace>可在rosrun时指定（见上文）
  * 句柄的命名空间会影响句柄下面订阅和发送的消息的命名空间

    ```bash

    ros::init(argc, argv, "listener");
    ros::NodeHandle nh1("~");
    ros::NodeHandle nh2("~foo");

    ...

    # sub1订阅的话题是 <node_namespace>/listener/topic1
    # sub2订阅的话题是 <node_namespace>/listener/foo/topic2
    ros::Subscriber sub1 = nh1.subscribe("topic1", ...); 
    ros::Subscriber sub2 = nh2.subscribe("topic2", ...);
    ```
    
---
<br>

## 1.4 ROS多线程
### 1.4.1 解决程序终止在 rospy.spin() 处的问题
https://blog.csdn.net/qq_30193419/article/details/100776075 
```python
import threading
 
def thread_job():
    rospy.spin()
 
if __name__ == "__main__": 
    # 代码...    
    add_thread = threading.Thread(target = thread_job)
    add_stread.start()    
    #剩余代码...
```
### 1.4.2 在callback func中调用 matplotlib 实时绘图的问题
* 问题起源：   
想在 callback 函数中用 matplotlib 画实时更新的图（图的数据由 msg 得到，没触发一次 callback 更新一次）
* 常规想法：  
用`plt.ion()`, 见：https://github.com/xianhu/LearnPython/blob/master/python_visual_animation.py 
  ```python
  import numpy as np
  import matplotlib
  import matplotlib.pyplot as plt
  import matplotlib.font_manager as fm
  from mpl_toolkits.mplot3d import Axes3D

  def three_dimension_scatter():
      # 生成画布
      fig = plt.figure()

      # 打开交互模式
      plt.ion()

      for index in range(50):
          # 清除原有图像
          fig.clf()

          # 生成测试数据
          point_count = 100
          x = np.random.random(point_count)
          y = np.random.random(point_count)
          z = np.random.random(point_count)
          color = np.random.random(point_count)
          scale = np.random.random(point_count) * 100

          # 生成画布
          ax = fig.add_subplot(111, projection="3d")

          # 画三维散点图
          ax.scatter(x, y, z, s=scale, c=color, marker=".")

          # 暂停
          plt.pause(0.2)

      plt.ioff()  # 关闭交互模式
      plt.show()  # 图形显示
      return
      
  three_dimension_scatter()
  ```
* 遇到的问题：  
如上所示，`plt.ion()`需要写在 ROS callback 函数外面，`fig.clf()`等则需要写在 callback 函数内部。ROS中的 callback 内部是一个子线程，但matplotlib作图所用的后端 `TKinter` 要求所有操作在一个线程内进行。所以报错 `main thread is not in main loop`，见：https://stackoverflow.com/questions/34764535/why-cant-matplotlib-plot-in-a-different-thread 
* 解决方案：  
Matplotlib不支持多线程，只支持多进程。将处理的数据作为一个话题输出，在另外一个ROS node中订阅并plot。





---
<br>

## 1.5 其他
### 1.5.1 rosbag和rosparam
* rosbag并不携带parameters信息，需要读取rosparam时，要单独操作
```bash
rqt_bag
rosbag record -a
rosbag play <filename>
rosparam dump <filename>
rosparam load <filename>
```
> 尽量减少dynamic parameters的使用，参见https://answers.ros.org/question/11008/best-practices-for-logging-messages-and-parameters/

---
<br>

## 1.5.2 ROS通信
Topic和Service的比较，参见：https://blog.csdn.net/ZXQHBD/article/details/72846865

---
<br>
