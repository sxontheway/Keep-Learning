## Efficient Video Processingon Embedded GPU
https://gstconf.ubicast.tv/protected/videos/v12586e9bc53dgvct23vir6pu9al37/attachments/213356.pdf

---
<br>

## 色彩编码
色彩编码大体上分为YUV和BGR两类，一般来说YUV类多用于传输，因为其所需的带宽比BGR编码的小。并且不需要三个通道同时传输，只有Y通道就能还原图像，只不过是黑白的 

YUV类：YUV可以有4:4:4, 4:2:2(UYVY等), 4:2:0(I420，NV12等)三种采样方式。参见：https://www.cnblogs.com/azraelly/archive/2013/01/01/2841269.html

YUV是色彩编码，和视频压缩编码是两个独立的东西。但摄像头如果说它是YUV输出，一般代表输出是YUV编码的Raw Video，而不是MJPEG等压缩过得视频流。

---
<br>

## 视频压缩编码
大体上分4派：MJPEG，MPEG-1/2/4, 国际电联的H.264/H.265, Google的VP8/VP9  

MJPEG：Motion-JPG，只有帧内的JPG压缩，无帧间压缩  
MPEG-1/2/4：帧内JPEG压缩+帧间压缩  
同等码率下，编码的视频质量：MJPEG < MPEG-2 < H.263 < MPEG-4(又叫mp4) < H.264(又叫AVC） < H.265(又叫HEVC) 

参考数量级： MJPEG: 1/20，MPEG-2: 1/40，MPEG-4: 1/70，H.264: 1/100  

VP8, VP9: https://www.zhihu.com/question/21067823

---
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

`omx`: openmax的缩写。Openmax是开放多媒体加速层（英语：Open Media Acceleration，缩写为OpenMAX），一个不需要授权、跨平台的软件抽象层，
以C语言实现的软件接口，用来处理多媒体。它由Khronos Group提出，目标在于创造一个统一的接口，加速大量多媒体资料的处理。
也即硬件厂商生产的硬件各不相同，但他们可以通过openmax这一个中间层使得用户接触到相同的api，用于多媒体数据的处理（视频等)  

`NVMM`: 非易失性主内存，Non Volatile Main Memory， 也即断电之后短时间内不会丢失  

`ximagesink`和`xvimagesink`: 见 https://blog.csdn.net/jack0106/article/details/5592557  

`nvvidconv`和`videoconvert`：
nvvidconv is a Gstreamer-1.0 plug-in which allows conversion between OSS (raw) video formats and NVIDIA video formats. 例如：
```
gst-launch-1.0 nvcamerasrc ! 'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)I420' ! nvoverlaysink -e
```
nvvidconv相当于一个桥梁，前边是"I420"类型过滤器，后边是BGRx类型过滤器。videoconvert同理，但是区别是 **nvvidconv有Nvidia 硬件加速，videoconvert则是在CPU上运行，没有加速，见 https://developer.nvidia.com/nvidia-video-codec-sdk**

## On Jetpack4.2: 
The above command works for Jetpack 3.3, change `nvcamerasrc`->`nvarguscamerasrc` and change `I420` -> `NV12` when using Jetpack 4.2.

---
<br>

## 关于TX2
TX2的GPU是pascal架构，pascal之后是最新的turing  
TX2的板载的摄像头是OV5693

<br>

## Gstreamer pipelines on Jetson TX2
> 官方文档：[ACCELERATED GSTREAMER FOR TEGRA X2 USER GUIDE](https://developer.download.nvidia.com/embedded/L4T/r28_Release_v1.0/Docs/Jetson_TX2_Accelerated_GStreamer_User_Guide.pdf?WVsbP1jiU5zK7ALWD3CN2SG2B6AqhZelh1cDn5CVNFnQMT8tK50S-MrbuUHKQmhD5zg6GOucEAxUPlr8BbrVWNElvDXoMRMkyMRCMM2ONjNaeXBJDMnRQbrh0v997n1O_V_BlpmvMLgtA-mQRSueIpqppyJt4sMacTZg4GaDihcpD5wMwBlmaxMNGxK0yiEeMw)

### 几个插件
* v4l2src  
参见：https://blog.csdn.net/jack0106/article/details/5592557  
是gstreamer给linux的一个插件，一般只用于video capture（也即decode）的功能

* nvcamerasrc  (In Jecpack4.2, change to nvarguscamerasrc)  
参见 https://developer.ridgerun.com/wiki/index.php?title=Gstreamer_pipelines_for_Jetson_TX2  
是Nvidia公司写的一个gstreamer插件，用于TX2上可以获得比v4l2src更低的CPU占用率

* nvgstcapture-1.0  
nvgstcapture-1.0 is a program included with L4T that makes it easy to capture and save video to file. It’s also a quick way to pull up the view from your camera.
This is an application based on gstreamer and omx to capture, encode and save video to the filesystem.  
这是一个程序了，基于gstreamer和v4l2src的程序，而不像v4l2src/nvcamerasrc是gstreamer的一个插件

<br>

### gstreamer在TX2上的具体实现
> 主要参考了 http://petermoran.org/csi-cameras-on-tx2/ 的ROS部分，对其gstreamer pipeline的部分进行修改即可  
> 需要修改的部分见 ***[gstreamer_code.md](./gstreamer_code.md)***  

用nvcamerasrc实现 (In Jecpack4.2, change to nvarguscamerasrc):  
除了最后一步其余都在NVMM上，节省了NVMM到standard memory之间的内存拷贝，也可以比v4l2src实现少用一个nvvidconv  

用v4l2实现:  
```
v4l2src device=/dev/video0 ! 
  video/x-raw, format=(string)I420, width=(int)1920, height=(int)1080 !
  videorate drop-only=true ! video/x-raw, framerate=30/1 ! 
  nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420, width=(int)$(arg width), height=(int)$(arg height) ! 
  nvvidconv ! video/x-raw, format=(string)BGRx ! 
  videoconvert ! video/x-raw, format=(string)BGR
```

* 用到的插件
  * videorate: (I420, 1920\*1080\*80fps) -> (I420, 1920\*1080\*30fps)  
  * nvvidconv1: (I420, 1920\*1080\*30fps) -> (I420, 960\*540\*30fps, NVMM) 
  * nvvidconv2: (I420, 960\*540\*30fps, NVMM) -> (BGRx, 960\*540\*30fps)  
  * videoconvert: BGRx -> BGR
  
* 说明
  * videorate的参数drop-only=true不能省掉，`framerate=30/1`要单独写，不与height，width等写一起  
  * videorate需要写在nvvidconv前面  
  * drop videorate是因为摄像头的输出分辨率是成对设置的(具体数值需参考摄像头文档)。例如对某CSI camera，要输出4k，只能是30fps；要输出1080p，只能是80fps。这时要想输出540p/30fps，只能在pipeline中手动更改分辨率和帧率
  * 两个nvvidconv不能合并，因为有一个从`普通内存->NVMM->普通内存`的过程

<br>

### NVMM的使用
参见: [关于 NVMM 和 nvvidconv 的讨论](https://devtalk.nvidia.com/default/topic/1012417/jetson-tx1/tx1-gstreamer-nvvidconv-will-not-pass-out-of-nvmm-memory/post/5162187/#5162187)  
* `gst-launch-1.0 v4l2src device=/dev/video0 ! "video/x-raw, format=I420(memory:NVMM)" ! nvvidconv ! nvoverlaysink -e`  
报错"could not link v4l2src0 to nvvconv0"，也即v4l2src只能写入到普通内存中，不能直接写入NVMM  

* `gst-launch-1.0 nvcamerasrc ! "video/x-raw, format=I420(memory:NVMM)" ! nvvidconv ! nvoverlaysink -e`   
    可行。因为nvcamerasrc插件直接将raw video写进NVMM了，nvvidconv要求input/output中至少有一个是NVMM（可以两个都是；只有一个是时有memoey copy的过程）
