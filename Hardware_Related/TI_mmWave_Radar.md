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

<br>

# 2. AWR1443 mmWave Demo 
https://github.com/juruoWYC/AWR1443-Demo-Analysis

<br>

# 3. DCA1000EVM+AWR1843BOOST
> 主要参考：https://e2echina.ti.com/blogs_/b/the_process/archive/2019/01/09/dca1000evm
中间遇到的一些bug及解决:
* 需要安装driver，若失败：可能是因为没禁用数字签名
* mmWave Studio不能顺利打开：需要在安装mmWave Studio之前安装matlab runtime (必须是 8.5.1 32 bit)
* mmWave Studio中SPI connectivity status不变绿（连接失败）：先用uniflash进行format
* 无法连接到FPGA：用某些usb hub转网口会出问题，最好将网线直接连在电脑上（用同typec也许可以，没试过）；USB，网线等需要先于电源接上
* RS232对应的COM口对应的XDS110 UART的Port
* Radar AWR上面的SOP应该设置成110，uniflash的时候是101，standalone的时候是100; AWR1843的S2应该拨到SPI，而不是CAN
* DCA1000EVM的SW2的key 5，设置成1(Software_config); Key3: 1243支持3发射天线，1642只支持2发射天线；Key2：可选择是存SD还是通过eternet发送
