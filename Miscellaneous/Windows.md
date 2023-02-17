# Vscode
* github 网页上可以按 `>` 进入 online vscode
* 在 windoes 中运行 python
    * 不同于linux，可以尝试 `python`，`py`，例如 `py a.py`

* Windows 上用 Vscode 配置 C  
   * Windows 10上使用vscode编译运行和调试C/C++，需要下载compiler，例如 `MingW`，见：https://zhuanlan.zhihu.com/p/77645306  

* Windows 上深度学习环境的搭建：cuda + cudnn + conda + pytorch
   * 下载安装 CUDA Toolkit，选择默认安装（会重装显卡驱动）
   * 下载 CuDNN，要与 CUDA 版本一致
   * 将 `cudnn-11.4-windows-x64-v8.2.2.26/cuda/bin` ，`xx/lib`, `xx/include` 中的内容复制到 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4` 中的对应目录
   * 添加 **系统的** 环境变量： 
      * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
      * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64
   * 检查安装结果：`nvidia-smi`
   * 下一步是下载安装 conda：https://www.anaconda.com/products/individual  
      安装过程中不要勾选替代系统 python 环境之类的
   * 添加 **系统的** 环境变量：`E:\anaconda3\Library\bin`
      * 这个命令不会让系统自动调用自动 conda 的 python，而需要通过在 shell 里面输入 conda activate base 才能启动
   * 紧接着安装 pytorch，按照官网的 pip 命令就可以   

* Windows + Vscode 上用 bash
   > https://stackoverflow.com/questions/42606837/how-do-i-use-bash-on-windows-from-the-visual-studio-code-integrated-terminal
   * 需要先安装 git bash
   * 安装后进入 VScode: `ctrl+shift+p --> Open Setting(Json) 打开 settings.json`
   * 加一行：`"terminal.integrated.shell.windows":  "D:\\git\\Git\\bin\\bash.exe"`（或其他安装路径），重启 Vscode
   * terminal 默认的 shell 就变为 bash 了，如果要用 powershell 或 cmd，在 terminal 中输入 powershell 或 cmd 即可

* Windows + VSCode + latex
    * 下载textlive，特别需要注意的一点是：`需要用管理员权限安装`，否则会报错；Vscode 中的 setting.json 的配置见 https://shaynechen.gitlab.io/vscode-latex/  
    * 用Texlive，file 'slashbox.sty' not found：https://blog.csdn.net/u010801696/article/details/79410545  

* 两台 Windows 传共享数据: 假设电脑A要传数据给电脑B  
   > https://consumer.huawei.com/cn/support/content/zh-cn00688400/ 
   * B开热点，A连B的 wifi，在A上用 ifconfig 查看自己的 ip
   * 在 A 上对想要共享的文件夹点右键：give access to
   * 在B上：`win+r`, `\\192.168.XXX.XXX` 即可访问A上共享的文件夹

# Others
* Visio 形状搜索始终找不到
    * Visio 语言首先需要和 Windows 系统语言一致
    * Control panel -> Indexing Options (得出现visio content) -> Advanced -> Rebuild
    
* Windows挂VPN并开热点
    * change adapter options -> 找到VPN那个connection -> properties -> sharing -> Allows sharing 并选择开热点对应那个connection


# QT 打包程序
> https://bkfish.github.io/2019/01/02/Qt%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E4%BB%A5%E5%8F%8A%E8%87%AA%E5%8A%A8%E6%89%93%E5%8C%85/ 

* 下载安装：http://download.qt.io/official_releases/qt/  
例如可以下载 qt-opensource-windows-x86-5.9.0.exe   
* 编译运行
* 打包：开始菜单打开 `QT-5.9 for Desktop (MinGW5.3.0)`等等，输入命令 `windeployqt <your_program>.exe`
