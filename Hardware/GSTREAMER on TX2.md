## 色彩编码
色彩编码大体上纷纷YUV和BGR两类，一般来说YUV类多用于传输，因为其所需的带宽比BGR编码的小。  

YUV类：I420, NV12, YV12, NV21，参见：http://blog.chinaunix.net/uid-28458801-id-4638708.html 

<br>

## 视频编码
大体上分3派：MPEG2/4, 国际电联的H.264/H.265, Google的VP8/VP9  

同等码率的前提下，各种标准的视频质量：MPEG2 < H.263 < MPEG4(又叫mp4) < H.264(又叫AVC） < H.265(又叫HEVC) 

数量级： MPEG-2: 1/25, H.264: 1/100  

VP8, VP9: https://www.zhihu.com/question/21067823

<br>

## 一些术语
`VLC`: 是将视频文件转换成流媒体的播放器。流媒体传输前先要encode，反之接受视频的第一步是decode  

`codec`: TX2的GPU有硬件加速decoding的单元  

`videotestsrc`: 这个element生成一个固定的video输出（通过pattern属性来设置），可用来测试视频的pipeline，
见 ***[Gstreamer cheat sheet](http://wiki.oz9aec.net/index.php/Gstreamer_cheat_sheet)*** 。例如:
`gst-launch-0.10 videotestsrc ! ffmpegcolorspace ! autovideosink`

 `ffmpegcolorspace`：视频适配的element，可以把一个色彩空间转换到另一个色彩空间（比如从RGB转到YUV）。它也可以在转换不同的YUV格式或者RGB格式。
当上游element和下游element是兼容的时候，这个element就是直通的，所以对性能的影响几乎是不存在的。

`capsfilter`: 当我们编程实现一个pipeline时，caps过滤通常用capsfilter这个element来实现。这个element不会修改数据，但会限制数据的类型。例如：
`gst-launch-0.10 videotestsrc ! video/x-raw-gray ! ffmpegcolorspace ! autovideosink`, 
其中'video/x-raw-rgb'这种写法就相当于是GstCapsFilter插件，

omx是openmax的缩写。Openmax是开放多媒体加速层（英语：Open Media Acceleration，缩写为OpenMAX），一个不需要授权、跨平台的软件抽象层，
以C语言实现的软件接口，用来处理多媒体。它由Khronos Group提出，目标在于创造一个统一的接口，加速大量多媒体资料的处理。
也即硬件厂商生产的硬件各不相同，但他们可以通过openmax这一个中间层使得用户接触到相同的api，用于多媒体数据的处理（视频等)  

`NVMM`: 非易失性主内存，Non Volatile Main Memory， 也即断电之后短时间内不会丢失  
`gst-launch-1.0 v4l2src device=/dev/video0 ! "video/x-raw, format=I420(memory:NVMM)" ! nvvidconv ! nvoverlaysink -e` 
报错"could not link v4l2src0 to nvvconv0"，也即v4l2src不能直接写入NVMM

`nvvidconv`和`videoconvert`：
nvvidconv is a Gstreamer-1.0 plug-in which allows conversion between OSS (raw) video formats and NVIDIA video formats. 例如：
 ```
 gst-launch-1.0 nvcamerasrc ! 
 video/x-raw(memory:NVMM), width=1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1 !
 nvvidconv flip-method=2 !
 video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink
 ```
nvvidconv相当于一个桥梁，前边是"I420"类型过滤器，后边是BGRx类型过滤器。videoconvert同理，但是区别是 **nvvidconv有Nvidia优化过的硬件加速，videoconvert没有**

<br>

## 关于TX2
TX2的GPU是pascal架构，pascal之后是最新的turing  
TX2的板载的摄像头是OV5693

<br>

## Gstreamer pipelines for Jetson TX2
> 参见：***[ACCELERATED GSTREAMER FOR TEGRA X2 USER GUIDE](https://developer.download.nvidia.com/embedded/L4T/r28_Release_v1.0/Docs/Jetson_TX2_Accelerated_GStreamer_User_Guide.pdf?WVsbP1jiU5zK7ALWD3CN2SG2B6AqhZelh1cDn5CVNFnQMT8tK50S-MrbuUHKQmhD5zg6GOucEAxUPlr8BbrVWNElvDXoMRMkyMRCMM2ONjNaeXBJDMnRQbrh0v997n1O_V_BlpmvMLgtA-mQRSueIpqppyJt4sMacTZg4GaDihcpD5wMwBlmaxMNGxK0yiEeMw)***  
* v4l2src
参见：https://blog.csdn.net/jack0106/article/details/5592557  
是gstreamer给linux的一个插件，一般只用于video capture（也即decode）的功能
* nvcamerasrc：
参见 https://developer.ridgerun.com/wiki/index.php?title=Gstreamer_pipelines_for_Jetson_TX2  
是Nvidia公司写的一个gstreamer插件，TX2板载摄像头OV5693一般用nvcamerasrc，其他摄像头可能在TX2shang用不了nvcamerasrc，只能用v4l2src  
* nvgstcapture-1.0  
nvgstcapture-1.0 is a program included with L4T that makes it easy to capture and save video to file. It’s also a quick way to pull up the view from your camera.
This is an application based on gstreamer and omx to capture, encode and save video to the filesystem.  
这是一个程序了，基于gstreamer 和v4l2src的程序，而不像v4l2src/nvcamerasrc是gstreamer 的一个插件

<br>

## 在TX2上用gstreamer启动CSI camera
* on-board camera OV5693 (using nvcamerasrc)
  见 http://petermoran.org/csi-cameras-on-tx2/  
  输出到屏幕：
  ```
  gst-launch-1.0 nvcamerasrc ! 'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)60/1' ! 
  nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)I420' ! nvoverlaysink -e
  ```
  用于ROS的launch文件：https://github.com/peter-moran/jetson_csi_cam/blob/master/jetson_csi_cam.launch

* Others (using v4l2src)
  输出到屏幕
  ```
  gst-launch-1.0 v4l2src device=/dev/video0 !  video/x-raw, width=\(int\)1920, height=\(int\)1080, format=\(string\)I420 ! 
  videorate drop-only=true ! video/x-raw, framerate=60/1 ! 
  nvvidconv ! nvoverlaysink overlay-w=1920 overlay-h=1080 sync=false
  ```
  用于ROS的launch文件：***[jetson_csi_cam.launch](./jetson_csi_cam.launch)***

