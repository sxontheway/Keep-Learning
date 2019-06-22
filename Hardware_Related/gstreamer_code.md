# nvcamerasrc

for ROS launch file
```
  <!-- Define the GSCAM pipeline -->
  <env name="GSCAM_CONFIG" value="nvcamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), format=(string)I420, width=(int)$(arg width), height=(int)$(arg height) ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR" />
```
for terminal command
```
gst-launch-1.0 nvcamerasrc sensor-id=0 ! video/x-raw\(memory:NVMM\), format=\(string\)I420, width=\(int\)1280, height=\(int\)960 ! nvvidconv ! nvoverlaysink overlay-w=640 overlay-h=480 sync=false
```

<br>

# v4l2
## USB2.0 camera (max: 1080p/25fps):
### for ROS launch file
```
  <!-- Define the GSCAM pipeline -->
  <env name="GSCAM_CONFIG" value="v4l2src device=/dev/video0 ! video/x-raw, format=\(string\)I420, width=\(int\)1280, height=\(int\)960 ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420, width=(int)$(arg width), height=(int)$(arg height) ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR" />
```
### for terminal command (wo/w framerate drop)
```
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, format=\(string\)I420, width=\(int\)1280, height=\(int\)960 ! nvvidconv ! nvoverlaysink overlay-w=640 overlay-h=480 sync=false

gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, format=\(string\)I420, width=\(int\)1280, height=\(int\)960 ! videorate drop-only=true ! video/x-raw, framerate=1/1 ! nvvidconv ! nvoverlaysink overlay-w=640 overlay-h=480 sync=false
```

---


## CSI camera (eg. e-CAM132, > 1080p/80fps):
### for ROS launch file
```
  <!-- Define the GSCAM pipeline -->
  <env name="GSCAM_CONFIG" value="v4l2src device=/dev/video0 ! video/x-raw, format=(string)I420, width=(int)1920, height=(int)1080 ! videorate drop-only=true ! video/x-raw, framerate=$(arg fps)/1 ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420, width=(int)$(arg width), height=(int)$(arg height) ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR" />
```
### for terminal command
```
gst-launch-1.0 v4l2src device=/dev/video0 !  
video/x-raw, width=\(int\)1920, height=\(int\)1080, format=\(string\)I420 ! 
videorate drop-only=true ! video/x-raw, framerate=60/1 ! 
nvvidconv ! nvoverlaysink overlay-w=1920 overlay-h=1080 sync=false
```
---
## Onboard Camera OV5693
### for ROS launch file
> https://github.com/peter-moran/jetson_csi_cam/blob/master/jetson_csi_cam.launch
```
<!-- Define the GSCAM pipeline -->
<env name="GSCAM_CONFIG" value="nvcamerasrc sensor-id=$(arg sensor_id) ! video/x-raw(memory:NVMM),
  width=(int)$(arg width), height=(int)$(arg height), format=(string)I420, framerate=(fraction)$(arg fps)/1 ! 
  nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR" />
```
### for terminal command
> http://petermoran.org/csi-cameras-on-tx2/
```
gst-launch-1.0 nvcamerasrc ! 
'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)60/1' ! 
nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)I420' ! nvoverlaysink -e
```
