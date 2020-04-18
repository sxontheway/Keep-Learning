
---
# 1. 概念
## 1.1 进程与线程
### 1.1.1 多进程
在任务管理器中，可以看到有几百个进程（Process），几千个线程（Thread），数十万个句柄。一个任务就是一个进程，打开两个记事本就启动了两个记事本进程，打开一个Word就启动了一个Word进程。  
对计算机，因为CPU上的指令是顺序执行的，原则上一个CPU的核某一时刻只能运行一个进程。要同时运行多个进程，例如边听歌边修改ppt，就必须使用并发技术。实现并发技术相当复杂，**但通俗理解就是所有正在运行的进程轮流使用CPU，每个进程允许占用CPU的时间非常短(比如10毫秒)**，这样用户根本感觉不出来CPU是在轮流为多个进程服务。但实际上在任一时刻有且仅有一个进程占有CPU的一个核。
### 1.1.2  多线程和超线程：
进程的颗粒度太大，一个进程内部会包含多个任务，例如word一边要进行拼写检查，一边要接受用户输入。**这些进程内的这些“子任务”被称为线程。**  
四核八线程则是指使用了超线程技术 , 把一个物理核心模拟成两个逻辑核心。理论上要像八颗物理核心一样在同一时间执行八个线程，所以设备管理器和任务管理器中会显示出八个核心，但事实上并不是真正的八个物理核心。
不管怎样，**任一时刻，一个模拟出来的逻辑核心也只能处理一个线程。**
### 1.1.3 线程的切换
* 同一进程内的线程间切换比进程间的切换要快
* 进程是资源分配的最小单位，线程是cpu调度的最小单位；资源分配的速度慢于CPU的处理速度。

假设程序 A 包括 a，b，c 等多个子程序段（也即线程）。那么这里具体的执行就可能变成：  
> CPU开始执行程序A -> CPU加载上下文 -> 开始执行A的a小段 -> 执行A的b小段 -> 执行A的c小段 -> CPU保存A的上下文

这里 a，b，c 是更为细小的CPU时间段，共享了 A 的上下文，也即线程共享了进程的上下文环境。
因为CPU自身并不存储上下文，所以每次进程切换，都有上下文的调入，保存，调出。**一个最基础的事实是：CPU太快了，寄存器仅仅能够追的上他的脚步，RAM和别的挂在各总线上的设备完全是望其项背。**   线程的引入使细粒度更高的调度成为可能。

<br>

## 1.2 虚拟机的网络连接
  * 桥接网络时，**物理网卡和虚拟网卡在拓扑图上处于同等地位**，物理网卡和虚拟网卡就相当于处于同一个网段。  
  * NAT(网络地址转换)模式，虚拟机借助宿主机上网。局域网其他主机是无法访问虚拟机的，但虚拟机可以访问局域网的所有主机，因为**真实的局域网相当于 NAT 的虚拟网络的外网。**  
  > 在某些局域网下(例如CUHK 1X)，桥接模式会导致无法上网、无法SSH等问题（所在局域网路由不允许）。此时只能使用NAT模式。

<br>

## 1.3 网络协议端口
### 1.3.1 端口的作用
一个IP地址的端口可以有65536（即2^16）个端口，端口通过端口号来标记的。  
在Internet上，各主机间通过TCP/IP协议发送和接收数据包。但因为计算机运行时大多是多进程，那么目的主机应该把接收到的数据包传送给众多同时运行的进程中的哪一个呢？这个问题有待解决，端口机制便由此被引入进来。  
本地操作系统会给那些有需求的进程分配协议端口，端口其实就是队。操作系统为各个进程分配了不同的队，数据包按照目的端口被推入相应的队中，等待被进程取用。  
不光接受数据包的进程需要开启它自己的端口，发送数据包的进程也需要开启端口，这样，数据包中将会标识有源端口，以便接受方能顺利地回传数据包到这个端口。  
### 1.3.2 三次握手
* 第一次：TCP 连接中，由 client 向 server 发送一个服务请求，所使用的 port 号通常是随机选择大于 1024 数。**（因为0-1023一般被用作知名服务器的端口，被预定，如FTP、HTTP、SMTP等）** 其 TCP封包会将（且只将） SYN 旗标设定起来。这是整个联机的第一个封包；
* 第二次： 如果 server 接受这个请求的话 **（特殊的服务需要以特殊的 port 来进行，例如 FTP 的 port 21 ）** ，则会向请求端送回整个联机的第二个封包。其上除了 SYN 旗标之外同时还将 ACK 旗标也设定起来，并同时在本机端建立资源以待联机之需；
* 第三次：client 获得 server 第一个响应封包之后，必须再响应对方一个确认封包，此时封包只带 ACK 旗标（事实上，后继联机中的所有封包都必须带有 ACK 旗标）；
> 只有当 server 收到 client 的确认 ACK 封包（也就是整个联机的第三个封包）之后，两端的联机才能正式建立。这就是 TCP 联机的三次握手。  

> 三次握手后，client 通常用的是高于 1024 的随机取得的 port，server 的 port 由其为 client 提供的服务确定而定，例如 WWW 选择 80，FTP 则以 21 为正常的联机信道。
### 1.3.3 常见端口
HTTP 80，HTTPS 443，IMAP 993，SOCKS 1080，FTP 21，SSH 22，SMTP 25，DNS 53 等  
ROS默认11311（可输入 echo $ROS_MASTER_URI 查看）

<br>

## 1.4 Linux版本
* Linux桌面环境(图形界面)：
  * Ubuntu自家的的桌面：Unity
  * 其他：GNOME，KDE，MATE(例如Linux Mint Mate或树莓派上的Ubuntu Mate)，LXDE(轻量级)，Xfce
* Linux常见发行版（按软件包格式分）
  * 基于dpkg(Debian package)，也即Debian系：
    * 商业发行版：Ubuntu
    * 社群发行版：Debian，Kubuntu(使用KDE桌面的Ubuntu)，Xubuntu(使用Xfce的ubuntu)，Lubuntu(使用LXDE的ubuntu)，Linux Mint等
  * 基于RPM(Red-Hat Package Manager)，也即Red Hat系
    * 商业发行版：RHEL(Red Hat Enterprise Linux)
    * 社群发行版：Fedora(Red Hat测试版，稳定性较差，一般用于桌面系统)，CentOS(RHEL的社区克隆版，稳定性较好，可用于服务器)，PCLinuxOS等
> 各个发行版本，命令大部分相通，但包管理命令和一些其他的系统管理命令不同。
> Ubuntu，Kubuntu，Xubuntu，Ubuntu Mate等核心都是相同的，只是桌面环境和预设安装的软件的不同。

<br>

## 1.5 包和包管理器
* Conda，Anaconda，Miniconda 的区别？
  * Conda：一种通用包管理系统，旨在构建和管理任何语言和任何类型的软件。有`conda install`，`conda update`，`conda remove`等命令。   
  * Anaconda：一个打包的集合，里面预装好了conda、某个版本的python、众多packages、科学计算工具等等，就是把很多常用的不常用的库都给你装好了。
  * Miniconda：它只包含最基本的python与conda，以及相关的必须依赖项，减小存储空间占用。
* Conda，pip，apt-get的关系？
  * 三者都包管理器。pip 允许你在任何环境中安装 Python 包，而 Conda 允许你在 Conda 环境中安装任何语言包（包括 C 或 python），apt-get 则是 Ubuntu 独有的。
  * 如果想在一个已有系统快速管理python包，可选择pip，因为conda应该在conda环境中使用，而pip鼓励在任何环境中使用
  * 而如果，想让许多依赖库一起很好地工作（比如数据分析中的 Numpy，scipy，Matplotlib 等）那就应该用conda，conda很好地整合了包之间的互相依赖。

------
<br><br>

# 2. Linux的使用 (Ubuntu and Debian)
Linux常用命令 http://www.runoob.com/w3cnote/linux-common-command.html
## 2.1 网络相关
### 2.1.1 NetworkManager  
NetworkManager是为方便网络配置而开发的网络管理软件包，最直接的体现是Ubuntu系统左上角的网络连接菜单。
```bash
# 停用NetworkManager.service
sudo systemctl stop NetworkManager.service 
sudo systemctl disable NetworkManager.service
# 启用NetworkManager.service 
sudo systemctl enable network.service 
sudo systemctl start network.service
```
### 2.1.2 查看SSH登录日志  
  * `/var/log`文件夹存放了日志信息，其中`auth.log`是ssh登录日志，可以看到其他计算机ssh登录本机的信息。  
  * `/etc/logrotate.conf`文件中可以更改日志存放多久。
  * `grep "Failed password" /var/log/auth.log | awk '{print $11}' | sort | uniq -c | sort -nr | more`可以看到有多少在ip尝试暴力破解。

### 2.1.3 SSH不能登录原因排查
* Server端
可能未安装ssh server：`apt-get install openssh-server`  
检查`/etc/ssh/sshd_config/`中的PermitRootLogin是不是no
* Client端口
删除`~/.ssh/known_hosts`文件，重新建立公钥（可能因为Server端重装了系统而ip没变）

<br>

## 2.2 系统相关
### 2.2.1 文件权限
> Linux系统支持多人多任务，常常存在多个用户同时使用一个Linux系统的情况。如何进行访问控制呢？  
* 所有者、群组和其他人
  * 所有者：Linux系统为每个用户提供一个用户主目录，用户的主目录为/home/<username>，该目录下所有者为这个用户。每个Linux用户创建的文件的所有者都是自己。
  * 群组：一个群组可以包含若干个用户。可以使群组内的成员有某项权限，而群组之外的其他人没有该权限。
  * 其他人：不属于文件所有者或文件所属群组成员的用户，便都是其他人。当然，root是一个特殊的超级用户，该用户可以访问Linux机器上的所有文件。
 
* `ls -l`命令显示的`drwxr-xr-x`是什么意思？  
  * r代表可读、w代表可写、x代表可执行（在linux中可执行代表可以进入，例如.sh文件需要先获得 x 权限才能成为脚本）
  * 第1位表示文件类型：d是目录文件，l是链接文件，-是普通文件，p是管道
  * 第2-10位分别表示`所有者、群组、其他人`拥有的权限，`rwx`分别表示可读可写可执行，没有该项权限用`-`占位。

* `chmod 774 run.sh`是什么意思？  
如上，三位数从前向后依次是所有者、群组、其他。可读：4，可写：2，可执行：1。`7 = 4+2+1`表示可读可写可执行。其他以此类推。

* 初创文件及目录的权限？  
初创文件的最大权限是666，实际权限要减去掩码（例如666-002=664）。初创目录的最大权限是777，实际权限也需要减去掩码。文件也可以拥有执行权限（X权限），但需要人为指定。  
用`umask`查看掩码，用`umask abc`修改掩码。  


### 2.2.2 分区扩容
> Linux 主分区一般在 sda1 下面。同 Windows 一样，普通情况下只有连续的空间才能重新分配，合并分区等。
> 用 lvm 可实现非连续的分配，但需要将所有 sda 在一开始（刚刚格式化完成之后）就设为 lvm 模式。  
* 相关命令
  * `df -h`显示已经挂载的分区列表  
  * `sudo fdisk -l`列出整个系统内能够搜寻到的装置的分区   
  * `df /`找出磁盘文件名 (一般为 dev/sda1，然后在下一步中去掉1)  
  * `sudo fdisk /dev/sda`查阅根目录硬盘相关信息  
* 改变磁盘大小步骤
  * 用 Gparted Partition Editor 等分区软件将 swap 分区暂时移到末尾，方便凑成连续分区以扩容。
  * 将 swap 分区修改后(例如将其分区号从 sda2 修改为 sda3 之类的)，虽然在`sudo fdisk /dev/sda`中显示有 swap 分区，但开机极慢。且需要在 Bios 中选 boot 模式，选 advanced options for ubuntu，再选第一个才能缓慢开机。  
  这是因为 swap 分区并没有真正地被系统识别，需修改UUID，参考 http://www.linuxdiyf.com/linux/20747.html
```
①在Extend分区划出一块Logical分区
②sudo fdisk -l 查看分区设备号，这里我的是/dev/sda7
③sudo mkswap /dev/sda7 将sda7转为swap分区格式
④sudo swapon /dev/sda7

进行完上述操作之后reboot后用free命令查看，swap显示仍旧为0，这是因为/etc/fstab 里的值并没有更新
①sudo blkid -t TYPE=swap 查看swap的Label（UUID）
②vim /etc/fstab 修改swap的Label即可
③重启

最后，可以修改swappiness的值，内存比较大的情况下，建议修改为10
①sudo vim /proc/sys/vm/swappiness 将其改为10
②sudo vim /etc/sysctl.conf 最后加上一行vm.swappiness=10
```
  * 参数 swappiness 来控制物理内存和虚拟内存的比例，默认值为60，意思是说在内存使用40%的情况下就使用 swap 空间。如果为10 ，也即物理内存使用90%时候再用虚拟内存。为0则，优先使用物理内存。  
  具体值需要依据内存大小而定，内存很大（例如32g时），可以优先使用物理内存。用`cat /proc/sys/vm/swappiness` 查看当前 swappiness 数值。

------
<br><br>

## 2.3 重要文件
### 2.3.1 关于source.list (Debian)
> **Ubuntu的source.list文件可以类推**  

`/etc/apt/sources.list`是一个普通可编辑的文本文件，保存了软件更新的源服务器的地址。`sudo apt-get update`命令本质上是读取了`sources.list`这个文件进行了软件包列表信息的更新。  
`/etc/apt/sources.list.d/*.list`和`/etc/apt/sources.list`是一样的功能，但为在单独文件中写入源的地址提供了一种方式，通常用来安装第三方的软件。  
`source.list`文件内部包含了镜像的URL，以网易镜像为例：http://mirrors.163.com/debian/ ，里面包含了很多目录，其中：  
* `/dists/`目录包含发行版, 此处是获得 Debian 发布版本(releases)和已发布版本(pre-releases)的软件包的正规途径  
* `/pool/`包含`contrib/`，`main/`，`non-free`等。
  * main：包括符合`DFSG(debian free software guidelines)`的开源软件。
  * contrib：即使本身属于符合DFSG的开源软件，但是依赖闭源软件来编译或者执行。  
  * non-free：不属于自由软件范畴的软件。
main仓库被认为是Debian项目的一部分，contrib和non-free是为了用户的方便而维护和提供。这样分是因为 Debian 是非营利组织，有一套完善的软件管理方式。基于其对软件 free 度的一种坚持，对不同版权软件包的录入有一些限定。**如果想一直能够在Debian上安装闭源软件包，你需要添加contrib和non-free软件仓库。**   

> * source.list的修改方法参照：https://wiki.debian.org/SourcesList  
> * 修改`/etc/apt/sources.list`之后可能会运行下面两个命令进行更新升级：  
`sudo apt-get update`：从镜像源更新的软件包列表信息  
`sudo apt-get dist-upgrade`：进行发行版升级(从debian 8升级到debian 9之类的)  

