> 以下均在 ***Raspberry Pi3 ModelB*** 上实现

# 1. 系统安装
* 做系统盘  
遇到 sd 卡写保护：  
win+R > cmd > diskpart > list disk > select disk * > attributes disk clear readonly > 重新插入
* 安装系统  
用`Win32DiskImager(未尝试)/Etcher(尝试可行)`做安装包 > 安装系统(按next就行) > 对sd卡分区等
> 实测(Pi 3B)：  
https://ubuntu-mate.org/raspberry-pi/ 中的Ubuntu Mate镜像不能boot  
https://downloads.ubiquityrobotics.com/pi.html 中的***ROS pre-installed***的镜像(LXDE)可以boot

> 用U盘替代tf卡的方案：  
参见 https://www.jianshu.com/p/2ed5f1c6367b  
---
<br>

# 2. 32位系统下的操作
> 本节的操作基于上文***ROS pre-installed***的32位版本： https://downloads.ubiquityrobotics.com/pi.html  

* When the Raspberry Pi boots for the first time, it resizes the file system to fill the SD card, this can make the first boot take some time.  
* On a Pi3, our image comes up as a Wifi access point. The SSID is `ubiquityrobotXXXX` where XXXX is part of the MAC address. The wifi password is `robotseverywhere`. Once connected, it is possible to log into the Pi with `ssh ubuntu@10.42.0.1` with a password of ubuntu. If you connect up a keyboard and mouse enter the password ubuntu at the prompt. `In virtual machine, to successfully use ssh, please choose NAT mode instead of Bridge.`  
* If you are not running on one of our robots run `sudo systemctl disable magni-base` to ensure that our startup scripts get disabled. 
---
<br>

# 3. 64位系统下的操作
> 本节操作基于Debian 9的64位版本： https://github.com/bamarni/pi64  
## 3.1 概念
> * CPU一般有两种架构： ARM和x86架构。两个架构对应不同的指令集
> * 处理器架构也叫微结构，是一堆硬件电路； 指令集会决定微结构的一部分硬件设计（解码逻辑和执行单元），但不是全部。 同一指令集，每个人的硬件电路设计并不一定完全相同。

Arm架构：
   * armel： 32位，浮点数计算用fpu(float processing unit)，传计算所需的参数用普通寄存器传，也即soft-float
   * armhf： 32位，浮点数计算用fpu，传计算所需的参数用fpu中的浮点寄存器传，也即hard-float
   * armv8： 64位的arm都默认是hard float的，因此不需要hf的后缀  
    
x86架构：
  * 32位版本的指令集由Intel公司开发。 64位版本的指令集被称作x86-64(也称为x64，AMD64)，由AMD公司开发  
  
树莓派3B的CPU
  * 树莓派3B的CPU是`bcm2837`，是`armv8`架构的，支持64位系统。只是现在官方发布的系统都是32位的，也就是用的`armv7`的指令集编译的
  * 在树莓派3B上安装32位系统后，输入`uname -a`，显示有`armv7`，对应于32位架构； 输入`getconf LONG_BIT`，显示32  
  * 从第三方下载树莓派的64位系统，安装64位版本后，输入`uname -a`，显示`aarch64`，是64位架构
      * `armv8`使用了两种执行模式，`aarch32`和`aarch64`。顾名思义，一个运行32位代码，一个运行64位代码      
      * `aarch64`：`armv8`构架下的64位执行模式，包括该状态的异常模型、内存模型、程序员模型和指令集支持等
   
## 3.2 系统安装
下载镜像：  
Ubuntu 18.04见：https://wiki.ubuntu.com/ARM/RaspberryPi#Booting_generic_arm64_ISO_images  
Debian 9见：https://github.com/bamarni/pi64  

## 3.3 安装ROS
ROS melodic对应于Ubuntu18.04，Debian9，可从apt-get直接安装：
http://wiki.ros.org/melodic/Installation/Debian

## 3.4 安装EdgeX

* 安装依赖项
  ```bash
  sudo apt update
  sudo apt install build-essential git pkg-config curl wget libzmq3-dev
  ```

* 创建swap分区  
为了编译EdgeX，运行MongoDB等，创建一个2GB的SWAP分区
  ```bash
  sudo touch /tmp/theswap
  sudo chmod 600 /tmp/theswap
  sudo dd if=/dev/zero of=/tmp/theswap bs=1M count=2048
  sudo mkswap /tmp/theswap
  sudo swapon /tmp/theswap
  ```

* 安装go环境  
需要去查询最新的EdgeX需要什么go版本，否则会报错
  ```bash
  sudo tar -C /usr/local -xvf go1.11.5.linux-arm64.tar.gz
  cat >> ~/.bashrc << 'EOF'
  export GOPATH=$HOME/go
  export PATH=/usr/local/go/bin:$PATH:$GOPATH/bin
  EOF
  source ~/.bashrc
  ```

* 安装glide：go的包管理器  
URL https://glide.sh/get 默认最新版本，它会自动适配硬件。但对应于你硬件的最新版本可能还未释出，此时会报错，解决方案如下：  
输入 https://glide.sh/get > 下载脚本 > 修改脚本 > 放在自己服务器上 > 再curl    
例如 `curl personal.ie.cuhk.edu.hk/~sx018/get | sudo PATH=$PATH GOPATH=/usr/local/go sh`
  ```bash
  curl https://glide.sh/get | sudo PATH=$PATH GOPATH=/usr/local/go sh
  ```

* (不用docker，源码编译)安装EdgeX  
  > 安装步骤见 https://www.hackster.io/mhall119/running-edgex-on-a-raspberry-pi-d35dd5  
  > https://oranwind.org/-edge-running-edgex-on-a-ubuntu/  

  * build
  go get 用于一键下载、编译、安装基于go的代码库  
    * go build : 编译出可执行文件  
    * go install : go build + 把编译后的可执行文件放到`$GOPATH/bin`目录下  
    * go get : git clone + go install
    ```bash
    go get github.com/edgexfoundry/edgex-go
    cd ~/go/src/github.com/edgexfoundry/edgex-go
    make prepare
    make build
    ```

  * 安装并启动MongoDB
    ```bash
    sudo apt install mongodb-server
    systemctl status mongodb
    wget https://github.com/edgexfoundry/docker-edgex-mongo/raw/master/init_mongo.js
    sudo -u mongodb mongo < init_mongo.js
    ```

  * 运行EdgeX，测试ping，应该返回pong
    ```bash
    cd ~/go/src/github.com/edgexfoundry/edgex-go
    make run
    curl http://EdgeX_IP:48080/api/v1/ping
    ```
    > EdgeX_IP是host的IP，48080为edgeX发送数据的端口  

* (用docker，非源码编译)安装EdgeX  
  * 安装docker, docker-composer  
    * docker安装：https://docs.docker.com/install/linux/docker-ce/debian/  
    * docker-composer：用pip安装即可
  * 安装edgeX
  https://docs.edgexfoundry.org/Ch-QuickStart.html#running-edgex
  * 测试
    > 参见：https://github.com/HeathKang/test-edgexfoundry  
    
    * 第一步：需要`pip3 install pymodbus`  
    * 第二步：运行`python3 ./script.py`；这个脚本负责产生数据，并定义用`address=(host_IP, Port)`发送数据  
    * 第三步：按照`.yml`文件启动在docker中启动edgeX，接受数据。若发生 Problem with MergeList 错误，解决方法如下： 
      ```bash
       sudo rm -vf /var/lib/apt/lists/*
       sudo apt-get update
      ```
    * 第四步：`curl http://localhost:48080/api/v1/event`中的48080为`.yml`文件定义的edgeX数据端口，此步尝试读取edgeX数据

* EdgeX 用 MQTT 通信的 Demo
> 参见 https://oranwind.org/-edge-tou-guo-opencv-xie-qu-ying-xiang-chuan-song-dao-kafka/
  
---
<br>

# 其他
* 自带的Raspbian系统，初始用户名：pi，密码：raspberry  
`passwd`改用户密码，`sudo passwd`改root密码
