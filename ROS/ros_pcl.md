# 1. 数据类型
## 1.1 ROS中3种Point cloud类型
参见：http://wiki.ros.org/pcl/Overview
* [sensor_msgs::PointCloud](http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud.html)  
ROS定义的一种msg格式，只支持3-D
* [sensor_msgs::PointCloud2](http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html)  
ROS定义的一种msg格式，PCL也支持。是`sensor_msgs::PointCloud`的加强版。可表示任意n-D的数据，Point values可以为各种基本类型。 Point data is stored as a binary blob, its layout described by the contents of the "fields" array.
* [pcl::PointCloud< PointT >](http://docs.pointclouds.org/trunk/classpcl_1_1_point_cloud.html)  
PCL中的点云格式，和PointCloud2类似，也有头文件。  
> ***Instead of filling out the `PointCloud2` message fields directly, it's much more comfortable to fill out a `pcl::PointCloud<T>` message instead and publish it via `pcl_ros`, which will do the conversion to a `sensor_msgs::PointCloud2` automatically.***
  
## 1.2 选用3种中的哪种？
Each point cloud object type gives you information about the field names in a different way.  

If you have a `sensor_msgs::PointCloud` object, they're all floats. 
To find out their names, look at the elements of the `channels()` vector; each one has a name field.  

If you have a `sensor_msgs::PointCloud2` object, look at the elements of the `fields()` vector; 
each one has a name field and a datatype field. PCL has methods for extracting this information, see [io.h](http://docs.pointclouds.org/1.7.1/common_2include_2pcl_2common_2io_8h_source.html).  

If you have a `pcl::PointCloud<T>` object, you probably already know what type the fields are because you know what T is. 
If it's a topic published by another node that you didn't write, you'll have to look at the source for that node. 
PCL has methods for extracting this information, see [io.h](http://docs.pointclouds.org/1.7.1/common_2include_2pcl_2common_2io_8h_source.html).

------
<br>

# 2. The pcl_ros Utilities
需要`#include <pcl_ros/point_cloud.h>`，参见：http://wiki.ros.org/pcl_ros  
* This header allows you to publish and subscribe `pcl::PointCloud<T>` objects as ROS messages. 
These appear to ROS as `sensor_msgs/PointCloud2` messages, offering seamless interoperability with non-PCL-using ROS nodes. 
For example, you may publish a `pcl::PointCloud<T>` in one of your nodes and visualize it in rviz using a `PointCloud2` display. ***The publisher takes care of the conversion (serialization) between sensor_msgs::PointCloud2 and pcl::PointCloud<T> where needed.*** intCloud<T>
The old format `sensor_msgs/PointCloud` is not supported in PCL. 
> 即`#include <pcl_ros/point_cloud.h>`后，`pcl::PointCloud<T>`也可以作为ROS中的msg被直接发布和订阅了。

------
<br>

# 3. ROS nodes
参见： http://wiki.ros.org/pcl_ros?distro=kinetic  
ROS提供了封装好的节点，可以直接利用他们进行以下四者的相互转换：  
`.bag`, `.pcd(a Point Cloud Data File Format)`, `sensor_msgs/PointCloud2`, `sensor_msgs/Image`

------
<br>

# 4. 编写自己的PointCloud\<T>
> 参见： http://www.pclcn.org/study/shownews.php?lang=cn&id=286,   
>      https://blog.csdn.net/qq_15332903/article/details/65444811

## 4.1 Why
* 当诸如`pcl::PointXYZI`等类型不够用时（例如我们要定义一个点包含x,y,z,vx,vy,vz,az,ay,az），有两种方法：  
  * 自己定义`PointCloud<T>`
  * 直接对`sensor_msgs::PointCloud2`类型进行赋值  
* 但1.1中说过，`sensor_msgs::PointCloud2`的格式复杂，且点云以二进制存储，对其直接进行填比较麻烦（但也不是不可以，下面会讲）。所以一般是通过填写`PointCloud<T>`,让`pcl_ros`来进行序列化及发布：
  ```cpp
  #include "pcl_ros/point_cloud.h"

  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  ros::Publisher pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> > (topic, queue_size);
  pub.publish(cloud);
  ```
## 4.2 How
* MyPointType.h文件：
  ```cpp
  #ifndef MYPOINTTYPE_H
  #define MYPOINTTYPE_H

  #include <pcl/point_types.h>
  #include <pcl/point_cloud.h>
  #include <pcl/io/pcd_io.h>

  struct MyPointType    // 在之后就可以用 MyPointType 替代 PointCloud<T>
  {
    PCL_ADD_POINT4D;	
    // This adds the members x,y,z which can also be accessed using the point (which is float[4])

    float range;
    float velocity;
    int16_t intensity;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;	// 确保new操作符对齐操作
  }EIGEN_ALIGN16;		// 16位SSE(Streaming SIMD Extensions)字节对齐

  POINT_CLOUD_REGISTER_POINT_STRUCT(MyPointType,// 注册点类型宏
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, range, range)
    (float, velocity, velocity)
    (int16_t, intensity, intensity)
    )

  #endif
  ```
* 解释
  * 为什么要字节对齐？  
  一方面，我们的计算机硬件就是这么设计的，也就是说CPU在访问存储器的时候只能在某些地址处获取某些特定类型的数据。另一方面，因为CPU读取数据的时候不是一个一个读的，而是几个几个读的。这样的话，即使请求的是第2-3bytes的内容，CPU也一次抓取了4或8byte的内容。  
  字节对齐就是让数据按一定规律存储，否则会降低读取速度，从而影响计算效率。 参见： https://www.zhihu.com/question/23791224/answer/25673480
  * PCL_ADD_POINT4D是什么？
    ```cpp
    #define PCL_ADD_POINT4D \
      PCL_ADD_UNION_POINT4D \
      PCL_ADD_EIGEN_MAPS_POINT4D


    #define PCL_ADD_UNION_POINT4D \
      union EIGEN_ALIGN16 { \
        float data[4]; \
        struct { \
          float x; \
          float y; \
          float z; \
        }; \
      };
      ```

------
<br>


# 5. 编写自己的PointCloud2
Debug指南：https://www.cnblogs.com/gdut-gordon/p/9155662.html  
Python sensor_msgs.msg.PointCloud2() Examples: https://www.programcreek.com/python/example/99841/sensor_msgs.msg.PointCloud2  
一个示例： 
```
def array_to_pointcloud2(self, data):
        msg =  PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id   # frame_id告诉了ROS坐标怎样进行变换
        msg.height = 1
        msg.width = data.shape[0]   # 每一次发布的发布的数据包含多少个点

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('range', 12, PointField.FLOAT32, 1),
            PointField('velocity', 16, PointField.FLOAT32, 1),
            PointField('intesity', 20, PointField.FLOAT32, 1),
            PointField('class', 24, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = 0
        msg.is_dense = 1
        msg.point_step = 28   # 每个点有7个属性，每个属性占4个bytes
        msg.row_step = msg.point_step * data.shape[0]
        msg.data = np.asarray(data, np.float32).tostring()
        print(data.shape)
        return msg 

```