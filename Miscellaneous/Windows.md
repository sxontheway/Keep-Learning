* Visio 形状搜索始终找不到
    * Visio 语言首先需要和 Windows 系统语言一致
    * Control panel -> Indexing Options (得出现visio content) -> Advanced -> Rebuild

* SSH 连不上 or VSCode Remote连不上
   * 可能因为远程主机发过来的公钥和之前本机存的公钥不一致，出现：`Host key verification failed.`，解决方法： `ssh-keygen -R 你要访问的IP地址`

* Windows + VSCode + latex
    * 下载textlive，特别需要注意的一点是：`需要用管理员权限安装`，否则会报错
    * 用Texlive，file 'slashbox.sty' not found：https://blog.csdn.net/u010801696/article/details/79410545  

* Windows挂VPN并开热点
    * change adapter options -> 找到VPN那个connection -> properties -> sharing -> Allows sharing 并选择开热点对应那个connection

* 在 windoes 中运行 python
    * 不同于linux，可以尝试 `python`，`py`，例如 `py a.py`

* Windows+Vscode+C  
Windows 10上使用vscode编译运行和调试C/C++，见：https://zhuanlan.zhihu.com/p/77645306   
一些概念明晰：https://zhuanlan.zhihu.com/p/82977824