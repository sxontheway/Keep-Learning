# 整理
* Ubuntu 命令
   * `uname -a`: 输出操作系统信息 

   * `lsb_release -a`：查看Ubuntu版本  

   * `df -lh`:查看存储  

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

* Vscode
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

<br>

# 记一次服务器修复
## 起因
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

## 现象
在`make install`运行时报错（虽然后来在虚拟机上用上述脚本，顺利升级），之后`ls`等命令失效，ssh连不上。到机房插上显示屏后，输入user name不跳转到输密码界面，无奈重启，重启以后无法登录，显示 `kernel panic not syncing attempted to kill init`

## 原因
由于并没有手动删除任何软链接和原有的动态库文件，所以原有的`2.17`版本的文件都还在，于是分析是软链接出错：本来该指向`XXX-2.17.so`的动态库指向了`XXX-2.18.so`，但是由于install失败了，这些`XXX-2.18.so`的动态库其实是损坏的。

## 解决方案
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


## 其他未尝试的方案
> https://www.ancii.com/apbjqqpl/ 

重新通过rpm安装glibc等: `rpm -Uvh --root=/mnt/sysimage/ --force glibc-2.17-105.el7.x86_64.rpm`

## Lessons
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
