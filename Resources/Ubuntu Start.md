* Firefox cannot synchronize  
    * 原来是Firefox的附加组件管理器不见了，我们只需要重新安装上即可: https://blog.csdn.net/JohnYu024/article/details/79763300  
    * Add-on manager link: http://mozilla.com.cn/thread-343905-1-1.html



* Latex 安装：
    > 要加中文：`\usepackage[UTF8]{ctex}`即可：https://jdhao.github.io/2018/03/29/latex-chinese.zh/ 
    * Linux下， 安装vscode
    * sudo apt-get install texlive-full cjk-latex latex-cjk-chinese
    * 安装 Latex Workshop 插件 

    * Debug:
        * 有一篇的Bibtex的Library未找到，结果显示所有reference有错



* No Wifi Model  
    * `iwconfig`: see whether there is a model called wlan0. If not, it means that the network card driver is not installed.  
    * `sudo lshw -c network`: see the model and vendor oationf the network card. For example, `BCM43142 802.11 b/g/m, vendor: Broadcom Corportion`  
    * For Broadcast network card:`sudo apt-get install`, `sudo apt-get bcmwl-kernel-source`

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
