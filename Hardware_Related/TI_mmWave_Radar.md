# 1. mmWave的工作流程
* 烧写CCS(Code Composer Studio) Project, 使SOP2=1
  * 如果要自己实现对mmWave Board的底层的编程，需要用Code Composer Studio
  * 以IWR6843EVM为例，它板载有DSP和MCU。对应的，在TI Resource Explorer上有CCS Project for DSS/MSS，分别是给DSP/MCU的程序。
  将他们通过Uniflash烧写，对mmWave Board进行编程。
  * CCS Project控制了mmWave Board的Dataflow，也即对原始ADC数据的处理流程，以及Output怎样编码以便于UART输出
  * 例如`ti/mmwave_sdk_03_00_00_08/packages/ti/demo/xwr68xx/mmw/docs/doxygen/html/index.html`讲了在mmWave SDK Demo中，
  从UART端口输出的数据的具体TLV格式是怎样的，也即Pointcloud或Heatmap是怎样进行编码成串行数据的
  （mmWave Demo Visualizer用的就是mmWave SDK Demo的CSS Project）
  > 自己写CCS Project是为了改写CFAR算法、点云的clustering算法，以在DSP上完成这部分处理  
  > 但是当mmWave连接到TX2时，可以只传基本的Pointcloud的XYZ，velocity，intensity信息即可，剩下的在TX2上处理
* 在ROS中，根据不同CSS Project定义的不同UART输出格式，对从UART接受到的数据进行解码
  * 怎样在ROS中加Heatmap接口，见 [Heatmap in ROS](https://e2e.ti.com/support/sensors/f/1023/p/725262/2687507?tisearch=e2e-sitesearch&keymatch=ros%20heat#2687507)
  * https://github.com/zrmaker/ti_ros 中，则对应了 [People Tracking and False Detection Filtering Demo](http://dev.ti.com/tirex/#/All?link=Software%2FmmWave%20Sensors%2FIndustrial%20Toolbox%2FLabs%2F50m%20Outdoor%20People%20Tracking%20and%20False%20Detection%20Filtering%20-%2068xx%2FUser's%20Guide)
  中的输出格式
* 相同的CCS Project可以有不同的Chirp Configuration文件(.cfg)，以实现不同的探测距离、分辨率等
* 用roslaunch运行，移除跳线帽SOP2=0
> * 注意.cfg中的一些参数更改后，必须要按RST进行reboot（断电再开都不行），这些参数有dfeDataOutputMode，channelCfg，adcCfg，lowPower。详情见SDK User Guide。


<br><br>

# 2. AWR1443 mmWave Demo 
https://github.com/juruoWYC/AWR1443-Demo-Analysis

