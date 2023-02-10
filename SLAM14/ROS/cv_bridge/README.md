# 1. ROS 和 OpenCV 之间的格式转换(C++)
## 1.1 概述
![见Picture/cvbridge.png)](./cv_bridge.png "CvBridge")
* OpenCV中图像格式为`cv::Mat`  
* ROS中节点间传输、Rviz能显示的图像格式为`sensor_msgs/Image`  
* cv_bridge中定义了`CvImage`类，`CvImage`包含了`cv::Mat`。`sensor_msgs/Image`和`CvImage`之间可以互相转换。

<br>**namespace cv_bridge主要包含的内容：**  
```C++
namespace cv_bridge {

/****************************************************************************************
用cv:Mat (和header, encoding) 构造了CvImage类的对象，再用toImageMsg()成员函数得到待发布的内容
****************************************************************************************/
class CvImage   
{
sensor_msgs::ImagePtr toImageMsg() const;
public:
  std_msgs::Header header;
  std::string encoding;
  cv::Mat image;
};

/*******(*******************
从sensor_msgs/Image到CvImage
***************************/
typedef boost::shared_ptr<CvImage> CvImagePtr;
typedef boost::shared_ptr<CvImage const> CvImageConstPtr;

// Case 1: Always copy, returning a mutable CvImage
CvImagePtr toCvCopy(const sensor_msgs::ImageConstPtr& source,
                    const std::string& encoding = std::string());
CvImagePtr toCvCopy(const sensor_msgs::Image& source,
                    const std::string& encoding = std::string());
// Case 2: Share if possible, returning a const CvImage
CvImageConstPtr toCvShare(const sensor_msgs::ImageConstPtr& source,
                          const std::string& encoding = std::string());
CvImageConstPtr toCvShare(const sensor_msgs::Image& source,
                          const boost::shared_ptr<void const>& tracked_object,
                          const std::string& encoding = std::string());
                          
}
```

## 1.2 Publish: 从OpenCV格式到ROS格式
详细内容、WebCamera视频的实时发布参见 http://wiki.ros.org/image_transport/Tutorials/PublishingImages
```C++
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("camera/image", 1);     //camera/image是自定义的话题名称
  cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);     //CV_LOAD_IMAGE_COLOR为OpenCV自带的宏定义
  cv::waitKey(30);       //延迟30ms或等待键值
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();  
  //构造了类CvImage(std_msgs::Header(), "bgr8", image)，再用其toImageMsg()成员函数，得到msg(要发布的内容)

  ros::Rate loop_rate(5);
  while (nh.ok()) {
    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
}
```
`cv::waitKey(x)` does two things:
* It waits for x milliseconds for a key press on a OpenCV window (i.e. created from cv::imshow()). 
Note that it does not listen on stdin for console input. 
If a key was pressed during that time, it returns the key's ASCII code. 
Otherwise, it returns -1. (If x is zero, it waits indefinitely for the key press.)
* It handles any windowing events, such as creating windows with cv::namedWindow(), or showing images with cv::imshow().

For instance, cv::waitKey(25) will show you a window for 25ms and after 25ms, the window will close automatically. If cv::waitKey(25) is in a loop, then a video will play frame by frame, with each frame lasting 25 ms.  
A common mistake for opencv newcomers is to call cv::imshow() in a loop through video frames, without following up each draw with cv::waitKey(30). 
In this case, nothing appears on screen, because highgui is never given time to process the draw requests from cv::imshow().  


## 1.3 Subscribe：从ROS格式到OpenCV格式
A complete example of a node that draws a circle on images and republishes them. 
```C++
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

image_transport::Publisher pub;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)               //在回调函数中进行从msg到CvImage的转换
{
  cv_bridge::CvImagePtr cv_msg = cv_bridge::toCvCopy(msg, "bgr8");       
  cv::circle(cv_msg->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));    //cv_msg->image即为cv::Mat格式
  pub.publish(cv_msg->toRos());
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "processor");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  pub = it.advertise("image_out", 1);
  image_transport::Subscriber sub = it.subscribe("image", 1, imageCallback);   
  ros::spin();
}
```
Encoding的格式：
* mono8: CV_8UC1, grayscale image
* mono16: CV_16UC1, 16-bit grayscale image
* bgr8: CV_8UC3, color image with blue-green-red color order
* rgb8: CV_8UC3, color image with red-green-blue color order
* bgra8: CV_8UC4, BGR color image with an alpha channel
* rgba8: CV_8UC4, RGB color image with an alpha channel 

>Note that **mono8** and **bgr8** are the two image encodings expected by most OpenCV functions. 


## 1.4 image_transport
image_transport目前只支持c++， 见 http://wiki.ros.org/image_transport  
image_transport专门用于ROS中图像的发布。使用它，可灵活方便地发送各种画质、各种码率的图像，例如：
```
image_transport ("raw") - The default transport, sending sensor_msgs/Image through ROS.
compressed_image_transport ("compressed") - JPEG or PNG image compression.
theora_image_transport ("theora") - Streaming video using the Theora codec. 
```
### 1.4.1 Example of image_transport
Instead of：
  ```C++
  #include <ros/ros.h>

  void imageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    // ...
  }

  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe("in_image_topic", 1, imageCallback);
  ros::Publisher pub = nh.advertise<sensor_msgs::Image>("out_image_topic", 1);
  ```
Do:
  ```C++
  #include <ros/ros.h>
  #include <image_transport/image_transport.h>

  void imageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    // ...
  }

  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("in_image_base_topic", 1, imageCallback);
  image_transport::Publisher pub = it.advertise("out_image_base_topic", 1);
  ```
详细内容参见 http://wiki.ros.org/image_transport

------
<br><br>

# 2. Python3下cv_bridge的导入
>方法基于： [Unable to import cv2 and cvbridge in python3?](https://github.com/ros-perception/vision_opencv/issues/196)

主要步骤：
1. 安装依赖项： `sudo apt-get install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-kinetic-cv-bridge`
1. 进入工作空间所在的文件夹，假设为`～/catkin_workspace/src`
1. 将`cv_bridge.zip`解压得到的cv_bridge文件夹放入`～/catkin_workspace/src`中
1. 将 [build.sh](./build.sh) 脚本放入`～/catkin_workspace`， 执行 [build.sh](./build.sh) 脚本
1. 在`.py`程序中，为导入python3版本的cv_bridge，使用和导入基于python3的cv2时一样的方法：
  ```bash
  #!/usr/bin/env python3
  # -*- coding:utf-8 -*-
  import sys, os
  
  ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
  if ros_path in sys.path:
      sys.path.remove(ros_path)
  cv_bridge_path = '/home/nvidia/Desktop/ti_ros/devel/lib/python3/dist-packages'    # Manually add cv_bridge path
  if cv_bridge_path not in sys.path:
    sys.path.append(cv_bridge_path)
  from cv_bridge import CvBridge, CvBridgeError
  from cv_bridge.boost.cv_bridge_boost import getCvType   # 若不报错，则基于python3的cv_bridge成功导入
  sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
  ```
  
注意事项：
* TX2等aarch64平台上，会没有`/usr/lib/x86_64-linux-gnu`这个文件夹，一种解决方案是在上面build（第4步）之前，手动创建软链接：
  ```bash
  mkdir -p /usr/lib/x86_64-linux-gnu
  sudo ln -s /usr/lib/aarch64-linux-gnu/libboost_python-py35.so.1.58.0 /usr/lib/x86_64-linux-gnu/libboost_python3.so
  sudo ln -s /usr/lib/aarch64-linux-gnu/libpython3.5m.so /usr/lib/x86_64-linux-gnu/libpython3.5m.so
  ```
  * 若操作失误，要删除软链接：`sudo rm -rf /usr/lib/x86_64-linux-gnu/libboost_python3.so`
* cv_bridge的源码来自于https://github.com/ros-perception/vision_opencv ，但根据[Unable to use cv_bridge with ROS Kinetic and Python3](https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3/50291787#50291787)， 需要将`src/vision_opencv/cv_bridge/CMakeLists.txt`中的：  
`find_package(Boost REQUIRED python3)`改为`find_package(Boost REQUIRED python-py35)`  
这是由于刚刚下载的`cv_bridge`这个包本身有点小问题。 按照`CMakelist.txt`引导Cmake会去找`libboost_python3.so`库， 但是在Ubuntu上只有`libboost_python-py35.so(/usr/lib/x86_64-linux-gnu/libboost_python-py35.so)`。

---
<br><br>

# 3. cv_bridge的使用（Python）
如上文所说， image_transport目前只支持c++， 但是python下可以订阅和发布CompressedImage类型的消息。 见 [Python CompressedImage Subscriber Publisher](http://wiki.ros.org/cn/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber)， 此教程中使用的是python2版本的cv2， python3版本的在命名上有一些区别，例如:  
需要将`cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)`改为`cv2.imdecode(np_arr, cv2.IMREAD_COLOR)`

## 3.1 ImpressedImage
* 注意使用imread， VideoCapture等读入图像时， 图像三个通道的顺序是BGR， 而不是RGB  
* 实测720p，25fps，CompressedImage带宽~3MB/s，RawImage带宽\~70MB/s。尽管如此，按下面代码的读取方式，CompressedImage：~0.02s/frame，RawImage：~0.003s/frame，也即通过cv_bridge读取RawImage 比用 numpy处理CompressedImage更快。
* 读取CompressedImage可能会损失一些处理的时间，但可以节省带宽和图像传输时间，因为RawImage的70MB/s的视频流会是一个瓶颈，使得RawImage这个话题的接收帧率低于发布帧率。
* 实测`img = resize(img, (60,80), anti_aliasing=False)`这一步比较耗时：原图尺寸720p，抗锯齿开~0.2s，抗锯齿关\~0.02s。

CompressedImage和RawImage消息的处理方法有不同。 RawImage是矩阵，CompressedImage是String(包含了图像的压缩信息），下面以捕获灰度图为例：
  * import部分差别不大
    ```python
    #!/usr/bin/env python3
    # -*- coding:utf-8 -*-
    
    import sys, os
    import rospy
    import numpy as np
    from std_msgs.msg import String
    from sensor_msgs.msg import Image, CompressedImage
    from skimage.transform import resize
    from skimage import img_as_ubyte

    ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
    cv_bridge_path = '/home/nvidia/Desktop/ti_ros/devel/lib/python3/dist-packages'

    # to import python3 version cv2 and cv_bridge
    if ros_path in sys.path:
        sys.path.remove(ros_path)
    import cv2
    if cv_bridge_path not in sys.path:
        sys.path.append(cv_bridge_path)
    from cv_bridge import CvBridge, CvBridgeError

    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
    ```
  * CompressedImage， 没用到cv_bridge
    ```python
    def callback(self, data):
        try:
            self.br = CvBridge()
            np_arr = np.fromstring(data.data, np.uint8)   # np_arr是ndarray格式，压缩的图片由一维向量表示
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # img是ndarray格式，(#row)*(#coloum)*(#channel)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # img是ndarray格式，(#row)*(#coloum)
        except CvBridgeError as e:
            print(e)
        img = resize(img, (60,80), anti_aliasing=False)	  # anti-aliasing 非常耗时
        img = img_as_ubyte(img)                           # skimage.transform.resize() 得到的是float
    ```
  * Raw Image， 用了cv_bridge
    ```python
    def callback(self, data):
        try:
            self.br = CvBridge()
            img = self.br.imgmsg_to_cv2(data,'mono8')     # img是ndarray格式，cv_bridge直接encoding成指定格式
        except CvBridgeError as e:
            print(e)
        img = resize(img, (60,80), anti_aliasing=False)	  # anti-aliasing 非常耗时
        img = img_as_ubyte(img)                           # skimage.transform.resize() 得到的是float
    ```
------
<br><br>
