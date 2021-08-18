# Vscode
* 在 windoes 中运行 python
    * 不同于linux，可以尝试 `python`，`py`，例如 `py a.py`

* Windows 上用 Vscode 配置 C  
   * Windows 10上使用vscode编译运行和调试C/C++，需要下载compiler，例如 `MingW`，见：https://zhuanlan.zhihu.com/p/77645306  

* Windows + Vscode 上用 bash
   > https://stackoverflow.com/questions/42606837/how-do-i-use-bash-on-windows-from-the-visual-studio-code-integrated-terminal
   * 需要先安装 git bash
   * 安装后进入 VScode: `ctrl+shift+p --> Open Setting(Json) 打开 settings.json`
   * 加一行：`"terminal.integrated.shell.windows":  "D:\\git\\Git\\bin\\bash.exe"`（或其他安装路径），重启 Vscode
   * terminal 默认的 shell 就变为 bash 了，如果要用 powershell 或 cmd，在 terminal 中输入 powershell 或 cmd 即可

* SSH 连不上 or VSCode Remote连不上
   * 可能因为远程主机发过来的公钥和之前本机存的公钥不一致，出现：`Host key verification failed.`，解决方法： `ssh-keygen -R 你要访问的IP地址`

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
