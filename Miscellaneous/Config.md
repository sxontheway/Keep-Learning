# 配环境相关操作

## Conda
```bash
# conda 安装 https://www.jianshu.com/p/edaa744ea47d
mkdir miniconda3
cd miniconda3
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 777 Miniconda3-latest-Linux-x86_64.sh #给执行权限
bash Miniconda3-latest-Linux-x86_64.sh -u #运行

# conda 内安装
conda create -n ml python=3.7
conda config --set auto_activate_base false
# 在 ~/.bashrc 最后一行加 conda activate ml

# 进入 (ml) 后
pip3 install matplotlib numpy scipy scikit-learn pandas pyyaml 

# 安装 nvidia driver 对应版本的 cuda/torch
nvidia-smi
nvcc --version  # 发现是 9.0 版本，只能安pytocrh 1.1
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
pip3 install tensorboard
```

## systemd
是 Linux 自带的系统工具，可以实现某些服务的自动重启

> https://www.ruanyifeng.com/blog/2016/03/systemd-tutorial-commands.html 
```bash
sudo systemctl daemon-reload  # 重载所有修改过的配置文件
sudo systemctl enable radar_record.service   # 开机自启动
sudo systemctl start radar_record.service    # 直接启动
sudo systemctl stop radar_record.service     # 停止
sudo systemctl disable radar_record.service  # 关闭自启动

# 查看 logger
sudo systemctl status radar_record.service   
journalctl -u radar_record.service -f
sudo systemctl is-active radar_record.service
sudo systemctl is-enabled radar_record.service
```

`/etc/systemd/system/radar_record.service` 文件：
```
[Unit]
Description=radar test service
After=multi-user.target

[Service]
Type=simple
User=root
RemainAfterExit=no
WorkingDirectory=/home/pi/radar_deploy
ExecStart=/usr/bin/python3 /home/pi/Desktop/collect_radar.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```


## Debug  
* Ubuntu 开机循环输入密码无法进入桌面的解决办法：  
`ctrl+Alt+f1~f6` 都是通过终端进入系统，`Ctrl+alt+f7` 是图形化界面进入，https://segmentfault.com/q/1010000007830779   

<br>

* Ubuntu 命令
   * `uname -a`: 输出操作系统信息 
   * `lsb_release -a`：查看Ubuntu版本  
   * `df -lh`：查看存储；`du -h --max-depth=1`：查看当前目录下文件夹大小
   * `sudo dmidecode`: 查看硬件配置，最常用的选项就是用`-t`来限定关键字，例如 `bios, system, baseboard, chassis, processor, memory, cache, connector, slot`  
   * `nvidia-smi`, `htop`, `nvtop`, `jtop`： 不同平台下的资源monitor 
   * 查看在GPU上的所有进程： `fuser -v /dev/nvidia*`
   * 批量清理GPU中的残留进程（例如孤儿进程）： `sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh`


<br>

* Firefox cannot synchronize  
    * 原来是Firefox的附加组件管理器不见了，我们只需要重新安装上即可: https://blog.csdn.net/JohnYu024/article/details/79763300  
    * Add-on manager link: http://mozilla.com.cn/thread-343905-1-1.html

<br>

* Latex 安装：
    > 要加中文：`\usepackage[UTF8]{ctex}`即可：https://jdhao.github.io/2018/03/29/latex-chinese.zh/ 
    * Linux下， 安装vscode
    * sudo apt-get install texlive-full cjk-latex latex-cjk-chinese
    * 安装 Latex Workshop 插件 
    * Debug:
        * 有一篇的Bibtex的Library未找到，结果显示所有reference有错
        
<br>

* No Wifi Model  
    * `iwconfig`: see whether there is a model called wlan0. If not, it means that the network card driver is not installed.  
    * `sudo lshw -c network`: see the model and vendor oationf the network card. For example, `BCM43142 802.11 b/g/m, vendor: Broadcom Corportion`  
    * For Broadcast network card:`sudo apt-get install`, `sudo apt-get bcmwl-kernel-source`

<br>

* Virtual box
   * 开不了机：可能是显卡的原因，关闭`3D加速` 
   * 双线剪贴板：`安装增强功能`
   * 共享文件夹：`sudo usermod -aG vboxsf $(whoami)`
   * 不能用USB 3.0：https://askubuntu.com/questions/783076/unable-to-enumerate-usb-device-under-virtualbox  
   只能连usb2.0口，或者用一个usb2.0的hub接在usb3.0口上
   * 在虚拟机中使用camera，`usb设备筛选器`中不要勾选usb camera，在`设备->摄像头`中勾选即可
      ```
      import cv2
      import numpy as np

      cap = cv2.VideoCapture(0)
      while(1):
          # get a frame
          ret, frame = cap.read()
          # show a frame
          cv2.imshow("capture", frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      cap.release()
      cv2.destroyAllWindows()
      ```
   * Note that most video capture devices (like webcams) only support specific sets of widths & heights.   
     Use `uvcdynctrl -f` to see. 

<br>

* Google Protobuf    
   > Protobuf 是由 Google 开发的一套开源序列化协议框架，类似于 XML/JSON，采用协议序列化用于数据存储与读取：https://blog.csdn.net/weixin_42730667/article/details/102550176  
  * Protobuf安装：  
  https://cloud.tencent.com/developer/article/1720879
  * `apt-get install` 时遇到 `apt-get /var/lib/dpkg/lock-frontend`相关的问题：  
  https://zhuanlan.zhihu.com/p/126538251  
  * 自定义一个 `.proto` 文件之后生成 `.h`，`.cc` 文件：  
  `protoc --cpp_out=./ ./BATsNet/radar_iwr6843/proto/radar.proto` 

<br>

* Vscode
   * Vscode Remote 的 `~/.ssh/config` 文件，直接复制的话可能会产生错误，一般手打一遍就好了

   * Vscode + jupyter-notbook:
      > https://stackoverflow.com/questions/60330837/jupyter-server-not-started-no-kernel-in-vs-code  
      * Press Command+Shift+P to open a new command pallete  
      * Type >Python: Specify local or remote jupyter server for connections -> default  
      * Type >Python: Select Intepreter to start jupyter notebook server
      
   * Pylint报错:
   ctrl+shift+p -> preference: open setting(JSON)  
   让pylint只显示error  
      ```json
      {
          "files.exclude": {
              "**/__pycache__": true
          },
          "python.linting.enabled": true,
          "python.linting.pylintArgs": [
              "--extension-pkg-whitelist=cv2", 
              "--disable=C,W"
          ],
      }
      ```
   * Vscode 不能 peek definition：
      ```
      -- utils
         -- utils.py
      -- yolov3
         -- utils
            -- utils.py
      -- train.py
      ```
      在`train.py`中`from utils.utils. import *`，但 import 的所有 functions 都不能 peek definition，是因为文件夹名称重复了、或文件夹名称和文件名称重复了，vscode不知道去找哪一个。

      * 对于那些安装在非标准位置的包（以tvm为例），在 `setting.json` 里面加 `"python.autoComplete.extraPaths": ["/home/xian/tvm/python"],`，然后 reload VsCode 即可

   * 使用 VSCode Debug 模式
      * 不能识别相对路径，需要在 `launch.json` 中配置工作目录
      * arguement 参数也需要在 `launch.json` 中进行输入
      ```json
      {
         "version": "0.2.0",
         "configurations": [
            {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd":"/home/xian/repos_FL/Few-shot_FL",
            "args": [
                  "--num_sample=5",
                  "--test_novel_only",
                  "--gpu=0"]
            }
         ]
      }
      ```
      * Vscode remote 意外断开，关闭程序：https://debug.fanzheng.org/post/vscode-remote-env-loading.html  

<br>

## Linux 下多 python 版本
已经有了 python2.7, python3.6，但想用 python3.8 的多进程间 share_memory 的功能，于是想要在 TX2 上安装 python 3.8  

* 安装 python 3.8: https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/  
   ```bash
   sudo apt update
   sudo apt install software-properties-common
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt install python3.8
   python3.8 --version
   ```

*  使用 update-alternatives
   ```bash
   # 查看路径
   which update-alternatives     # 返回 /usr/bin/update-alternatives

   # 添加 py3.8 到 update-alternatives
   sudo /usr/bin/update-alternatives --install /usr/bin/python python /usr/bin/python3.8

   # 切换版本
   sudo /usr/bin/update-alternatives --config python
   ```


## Vscode + Remote + X11 Forwarding
### 起因
尝试在 TX2 上安装了 VScode，但是键盘按键有延迟，于是放弃；但是有可视化需求，故打算对 TX2 配置 Remote + X11 Forwarding  
需求：在 Windows 10 PC 上用 Vscode remote，对 TX2 的窗口进行可视化

### 步骤
> https://blog.csdn.net/Dteam_f/article/details/109806294  
* 首先安装 Vscode 插件：
   * PC 上 Remote X11 (SSH)
   * TX2 上 Remote X11
* PC 上安装 X Server，推荐 XMing：https://zh.osdn.net/projects/sfnet_xming/downloads/Xming/6.9.0.31/Xming-6-9-0-31-setup.exe/  
* 配置使用密钥登录（免密码）、并开启 X11 Forwarding
   > https://blog.csdn.net/u010417914/article/details/96918562  
   > https://zhuanlan.zhihu.com/p/260189540  
   * 在 TX2 上
      ```
      # 建立密钥对
      ssh-keygen
      cd ~/.ssh
      
      # 确保生成功authorized_keys
      cat id_rsa.pub >> authorized_keys
      ls

      # 保证以下文件权限正确
      sudo chmod 600 authorized_keys
      sudo chmod 700 ~/.ssh

      # 打开SSH配置文件
      sudo vim /etc/ssh/sshd_config

      # 使用密钥登录
      RSAAuthentication yes
      PubkeyAuthentication yes

      # 开启 X11
      X11Forwarding yes
      X11UseLocalhost yes

      # 重启 SSH
      sudo service sshd restart
      ```
   * 在 PC 上  
   将 TX2 上 `/home/alex/.ssh/id_rsa` 文件复制到 `C:\Users\alex\.ssh` 文件夹中
   * 重启SSH，应该就可以免密码登陆了

* 更改 PC VScode 上 remote 相关配置  
   `.ssh/config` 文件：
   ```
   Host tx2
   HostName 192.168.xx.xx
   Port 22
   User alex
   ForwardX11 yes
   ForwardX11Trusted yes
   ForwardAgent yes
   ```
   `ssh_config` 文件：加上 `ForwardX11 yes`
*  在 TX2 上配置环境变量
   * 在 `~/.basshrc` 中加上 `export DISPLAY='localhost:10.0'`，并 `. ~/.bashrc`更新
   * `echo $DISPLAY` 输出10

* PC 上重启 VScode 并连接 reomte
* 打开 Xlaunch，选择 Display Number=10（和TX2的环境变量要匹配），其他默认
  * 后来试了试，即便`echo $DISPLAY` 输出10，`Xlaunch`中选0也可以成功显示
* 运行 `xeyes`，或`xclock`，或下面的 python 画图程序
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from mpl_toolkits.mplot3d import Axes3D
   ax = plt.axes(projection='3d')
   ax.scatter(np.random.rand(10),np.random.rand(10),np.random.rand(10))
   plt.show()
   ```

<br>



# 服务器运维

## Soft Lockup and Hard Lockup
> https://blog.csdn.net/Rong_Toa/article/details/109606560  

实验室serve崩了，症状：不能SSH，键盘连接到USB口指示灯不亮（USB口无供电），接上显示屏：`sending NMI from CPU 23 to CPUs31; Shutting down cpus with NMI; Hard LOCKUP`  
解决方案：强制重启  

<br>


## 记另一次服务器修复
### 起因
> https://juejin.im/entry/6844903765321973768  

想在Centos 7.2.1511的系统上装联邦学习的库 Pysyft。但是Centos的GLIBC是2.17的，Pysyft要求2.18的GLIBC，于是网上找了一个升级脚本：https://blog.csdn.net/wiborgite/article/details/87707938  
   ```bash
   curl -O http://ftp.gnu.org/gnu/glibc/glibc-2.18.tar.gz
   tar zxf glibc-2.18.tar.gz 
   cd glibc-2.18/
   mkdir build
   cd build/   
   ../configure --prefix=/usr
   make -j2    # 编译
   make install   # 安装
   ```

### 现象
在`make install`运行时报错（虽然后来在虚拟机上用上述脚本，顺利升级），之后`ls`等命令失效，ssh连不上。到机房插上显示屏后，输入user name不跳转到输密码界面，无奈重启，重启以后无法登录，显示 `kernel panic not syncing attempted to kill init`

### 原因
由于并没有手动删除任何软链接和原有的动态库文件，所以原有的`2.17`版本的文件都还在，于是分析是软链接出错：本来该指向`XXX-2.17.so`的动态库指向了`XXX-2.18.so`，但是由于install失败了，这些`XXX-2.18.so`的动态库其实是损坏的。

### 解决方案
* 到主机机房，用u盘制作相同操作系统的系统盘（CentOS-7-x86_64-Everything-1511），按F11（每个机器可能不同） -> troubleshooting -> 进入rescue模式。相当于把原来的系统挂载到救援系统的`/mnt/sysimage/`目录下
* 查看出错的软链接
   ```bash
   cd /lib64   
   ls -l /lib64/ | grep 2.17.so
   ls -l /lib64/ | grep 2.18.so
   ```
* 修复那20来个软链接
   > https://www.cnblogs.com/barneywill/p/10315603.html 
   
   例如：`ln -snf ld-2.17.so ld-linux-x86-64.so.2`


### 其他未尝试的方案
> https://www.ancii.com/apbjqqpl/ 

重新通过rpm安装glibc等: `rpm -Uvh --root=/mnt/sysimage/ --force glibc-2.17-105.el7.x86_64.rpm`

### Lessons
* 遇到这种问题，如果不能物理地接触主机（只能远程），那一定不要断开ssh，不要退出root权限（su命令也失效）
* `libc.so.6`是c运行时库，几乎所有程序都依赖c运行时库。程序启动和运行时，是根据libc.so.6 软链接找到glibc库。删除libc.so.6将导致系统的几乎所有程序不能工作，
* 命令
   * 打印动态库的依赖关系：例如`ldd /bin/ls`，`ldd /bin/pwd`.在实际linux开发与调试中，我们有时候需要对可执行bin移植，如果有动态链接的库，那么直接把bin移植过去，是无法直接运行的。我们需要把对应的依赖的动态库也一起移植，ldd命令就很方便的帮助我们查看，哪些依赖的动态库需要一起移植
   * `find`命令：`find ./ -name "*.jpg"`，查找本目录下所有jpg文件
   * `LD_PRELOAD` 的作用：是个环境变量，用于动态库的加载，它在动态库加载的中优先级最高。例如`LD_PRELOAD=/lib64/libc-2.17.so ln -snf /lib64/libc-2.17.so /lib64/libc.so.6`，可以在`libc.so.6`指向`libc-2.17.so`的软链接被误删除之后（几乎所有命令都无法执行时），重新建立软链接
   * `chroot /mnt/sysimage`，一个切换根目录的命令，在rescue时，可以直接从u盘系统切换到原系统上干活
* 查看版本
   * 查看Linux内核版本：`cat /proc/version `
   * 查看系统版本：`lsb_release -a`
   * 查看系统支持的glibc版本：`strings /lib64/libc.so.6 | grep GLIBC_`
   * 查看`libc.so.6`当前指向的版本：`ls -al /lib64/libc.so.6`
