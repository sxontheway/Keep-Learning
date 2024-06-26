# Bash
## 语法
1. 文件开头第一行`#!/bin/bash`，表示使用bash解释器
1. For 循环例子
    ```bash
    list=(1 2 3 4 5 6 8 10)
    for i in ${list[@]}
    do
        echo -e $i
        echo -e "$(( $i*2 ))\n"     # 用 $(()) 执行简单算数运算
    done

    for i in {1..5}
    do
        touch file$i.txt            # 直接将 $i 填到想要的位置
    done
    ```

<br>

## 符号
> https://www.cnblogs.com/balaamwe/archive/2012/03/15/2397998.html

* 其中  `$() ` 和` `` `(反引号)作用相同，是把括号内命令的输出再作为命令执行
* `${}`括号中放的是变量，`$()`中放的是命令
* 例子
    * `2>&1 | tee XXX.log`
        ```bash
        # 其中 2>&1 means "send any error messages (aka 'stderr') to the same output as any informational messages (aka 'stdout")." 
        # | tee XXX.log means "whatever output there is should also be sent to the file XXX.log"
        ```
<br>

## 常见命令
1. `cat`：连接文件
    * `cat test.txt`：打印整个文件到屏幕
    * `cat /dev/null > test.txt`：清空text.txt文件
    * `cat << EOF`:系统会提示继续进行输入，输入多行信息再输入EOF，中间输入的信息将会显示在屏幕上 
    * ```bash
      # 将输入的多行追加到test.txt中，EOF只是一个标示，可换成其他任意合法字符
      cat >> ~/.test.txt << 'EOF'
      aaaaa
      bbbbb
      EOF
      ```
1. `chmod`：变更文件或目录权限。例如`chmod a+x test.txt`：使所有用户都可以执行这个文件  
1. `chown`：变更文件拥有者。例如`chown root:root test.txt`：设置文件所有者和用户群组均为root
1. `cp`: 见 https://www.cnblogs.com/zdz8207/p/linux-cp-dir.html
1. `curl`： 利用URL语法在命令行下进行文件传输
    * `curl -sSL https://get.docker.com/ | sh`和`sh -c "$(curl -sSL https://get.docker.com/)"`  
    两者在功能上相同。但前者运用管道，curl和sh几乎同时执行；后者curl严格先于sh执行。  
1. `df -f`：查看磁盘空间大小
1. `echo`：输出字符串。若没有进行重定向（>>，>），则在终端中输出
1. `export`：将局部变量输出，成为全局变量。
    * 在一个terminal中用export，所定义变量的生命周期仅为该terminal内。要想该环境变量在之后的terminal中都生效，需要写在 ~/.bashrc 中
      ```bash
      export AAA=aaa
      export -p   #显示 declare -x AAA="aaa"
  
      #打开一个新的terminal
      export -p   #显示未定义AAA
  
      #在~/.bashrc中加一行：export AAA=aaa，对之后打开的每个terminnal都有
      export -p   #显示 declare -x AAA="aaa"
      ```
1. `find`： 按照一定搜索条件搜索文件或目录，最基本的查找是按全名或部分名称查找
    * 语法：`find   path   -option   [   -print ]   [ -exec   -ok   command ]   {} \`  
      例如，在 `/` 路径下找名字带有opencv的文件和文件夹，忽略大小写：`sudo find / -iname "*opencv*"`
1. `ln`：硬链接和软链接， 参见：https://www.cnblogs.com/xiaochaohuashengmi/archive/2011/10/05/2199534.html 
1. `locate`：在数据库中查找，比find快（find是在硬盘中查找），这个数据库需要手动更新维护（见updateb）
1. `grep`：Global Search Regular Expression and Print， 例如： `grep 'h.*p' /etc/passwd`  
1. `sh`：使用bash这个shell来执行脚本（shell也就是解释器）
    * `sh -c`让 bash 将一个字串作为完整的命令来执行，可以方便将 sudo 的影响范围扩展到整条命令
      ```bash
      echo "aaa" >> test.txt      # 成功执行
  
      # 更改文件所有者，使得只有root才能进行写操作
      sudo chown root:root test.txt
  
      # 用 sudo 并配合 echo 命令，尝试进行写操作
      sudo echo "bbb" >> test.txt     # Permission Denied，权限不够
      #  这是因为重定向符 > 和 >> 也是 bash 命令，sudo 只让 echo 命令有了 root 权限，> 和 >> 命令还是没有 root 权限，所以不能写入。
    
      # 解决方案：
      sudo sh -c 'echo "bbb" >> test.txt'
      ```
    * `sh filename`，`source filename`，`./filename`执行脚本的区别？
      * 当shell脚本具有可执行权限时，用sh filename与./filename执行脚本是没有区别得。 当脚本不具有可执行权限时，bash会把文件名当作命令，从而报错输出command not found.
      * sh filename 重新建立一个子shell，在子shell中执行脚本里面的语句，该子shell继承父shell的环境变量，但子shell新建的、改变的变量不会被带回父shell，除非使用export。
      * source filename：这个命令其实只是简单地读取脚本里面的语句依次在当前shell里面执行，没有建立新的子shell。那么文件脚本里面所有新建、改变变量的语句都会保存在当前shell里面。
  
1. `source`：1）使刚修改的文档立即生效　　2）将一个文件的内容当成脚本执行(首先需要配置x权限)
1. `tar`： 解压: `tar -zxfv XXX.tar.gz` 
1. `touch`：更新文件访问和修改时间或创建空文件
1. `unset`：用于删除已定义的shell函数和shell变量(包含环境变量)
1. `updateb`：用于创建或更新slocate/locate命令所必需的数据库文件，这个数据库中存放了系统中文件与目录的相关信息。 例如：
    ```bash
    sudo updateb  # 更新locate.db数据库
    locate eigen3  # 定位eigen3这个库
    ```
1. `wget`：和curl功能相似，但更偏重于下载的功能
    * `wget -c http://cn.wordpress.org/wordpress-3.1-zh_CN.zip`：断点续传，下载到当前目录

1. ：端口查看；`lsof -i`，`lsof -i:<Port_Num>`，`kill -9 <PID>`杀掉端口

<br>

## 其他
* 在 `bashrc` 中设置代理：  
`export http_proxy = http://proxyAddress:port`，  
`export http_proxy = https://proxyAddress:port`  
密码可能需要 encode to URL-encoded format 转义
* 挂载
    * 挂载硬盘 `mount -t nfs -o vers=3,nolock <efs_ip> <挂载路径>`

<br>

# Docker
## 背景：镜像和容器
* 为了节省资源，docker 是分层的，下层是镜像（read-only），上层是容器（read-and-write）  
上层的容器可能会复用同一个下层镜像，而容器和容器之间是独立的
* 运行中的镜像称为容器，你可以修改容器（比如删除一个文件），但这些修改不会影响到镜像。不过用 docker commit 可以把一个正在运行的容器变成一个新的镜像：
  * 例如 `docker commit --change "ENTRYPOINT" "/bin/bash" <container-id> <image-name>` 

## 查看 镜像/容器
* 查看镜像：`docker images`
* 查看容器状态：`docker ps -a`，不加 `-a` 就只查看运行中的，`-l` 看最近的
* 删除容器 `docker rm XX`，删除镜像 `docker rmi XX`

## 创建/启动/停止
* 创建并启动容器：`docker run -it --name pytest <Repository>:<TAG> /bin/bash`
    * 其中 `pytest` 是容器名
    * `<Repository>:<TAG>` 是镜像名，可以通过 `docker images` 查看；
    * `/bin/bash` 配合 `-it`，可以在容器中进入终端；`-i` 是交互模式，`-t` 是命令行
* 在 running container 中执行命令：`docker exec -it <container_name> bash`
* 启动/停止：`docker start/stop/restart <container_id>或<name>`，其中 `start` 可以加 `-i` 进入命令行
* 区别
    * docker run 只在第一次运行时使用，将镜像放到容器中，以后再次启动这个容器时，只需要使用命令 docker start 即可
    * docker run 相当于执行了两步操作：将镜像放入容器中（docker create），然后将容器启动，使之变成运行时容器（docker start）
    * docker start 的作用是，重新启动已存在的镜像。也就是说，如果使用这个命令，我们必须事先知道 `<container_id>或<name>`（用`docker ps`查看）
* GPU 容器：用新的 entrypoint 覆盖镜像的，用上 Nvidia runtime 和 GPU、宿主机挂载
    * 把 docker 默认的 runtime 替换成了 NVIDIA 自家的 nvidia-container-runtime，支持在容器中使用 GPU
    * 用自定义的 entrypoint 取代镜像中默认的；也就是容器启动后第一步的操作，比如原始镜像可能是 `python test.py`
    * `-dit`：`-d` 表示分离模式，容器会在后台作为一个守护进程运行，而不是直接退出；`-i` 允许对容器的 stdin 进行交互;`-t`是分配一个伪终端
        ```bash
        docker run --entrypoint bash -dit \
        --runtime=nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v /nas:/mnt  \
        --name pangu_sx registry-cbu.huawei.com/cloudbu-llm/pangu-38b-infer-strem:3.0.0
        ```
    * 启动容器：`docker start -it pangu_sx`

## 导入导出/保存载入
* 导出/导入 容器快照
    * `docker export container_ID > XXX.tar`
    * 下载一个 `rocketmq-3.2.6.tar.gz` 镜像，导入镜像并命名： `cat rocketmq-3.2.6.tar.gz | docker import - <Repository>:<TAG>`
* 保存/载入 镜像
    * `docker save -o rocketmq.tar rocketmq`：将 `rocketmq` 镜像（`docker images` 查看）保存成 `rocketmq.tar`
    * 载入镜像：`docker load < rocketmq.tar`，用法：将另外一台机器上拷贝过来的 .tar 读入成本机的镜像
* docker load-save / import-export 区别
    * `save/load`：对象是镜像，保存的是分层文件信息（联想docker的分层结构），导出的文件大
    * `docker export/import container_id`：对象是容器，保存的是容器当时状态的快照。是一个文件系统，丧失了分层结构

## 例子
### 把容器打包成一个镜像
* 先查看有哪些镜像：`docker images`
* 启动容器：
    * 例如，选定 `test:cuda10.2-torch1.12-ubuntu18.04-python3.8` 这个镜像（也可以用 Image ID），把宿主的 `/mnt/ssd_host` 挂载到镜像中的 `/mnt/ssd` 路径上，创建一个叫 `hello_world` 的容器，并进入命令行：  
        ```bash
        nvidia-docker run -it \
        -v /mnt/ssd_host:/mnt/ssd \
        --name hello_world test:cuda10.2-torch1.12-ubuntu18.04-python3.8 \ 
        /bin/bash
        ```
        * 其中 `nvidia-docker` 命令使得可以调用显卡，使用 `whereis` 和 `cat` 看具体做了什么
        * 文件挂载：`-v /test1:/test2`，是将 host 的` /test1` 目录挂载到容器的 `/test2` 目录

* 传文件进容器：`docker cp <本地文件路径> ID:<容器文件路径>`
* 在容器中配置好环境，调试好代码，例如 pip 等
* 将容器（80cdd11f9b60）打包保存成镜像，用于跨机器的读取
    > https://blog.csdn.net/github_38924695/article/details/110531410
    * 把容器打包成为镜像，使得在 `docker images` 中能看见  
        * `docker commit -a "eric" -m "my python test" 80cdd11f9b60  hello:v1`，其中 -a 提交的镜像作者，-m 提交时的说明文字，`hello:v1` 是镜像的 name 和 tag
    * 跨机器：本机上 docker export，另一台设备上 docker import（舍弃了分层结构，只作为新的基础镜像）


### 从别人的 .tar 加载自己的容器
* 从 .tar 文件导入/导出镜像：
    * `docker load -i xxx.tar`，会自动恢复 save 时的镜像名
    *  `docker save -o <PATH> <IMGAE_NAME>` 保存
* 图方便：不需要一个自己的镜像的话，不用修改镜像，而直接从原镜像构建容器，同时修改 entrypoint（加上 `--entrypoint bash`）  

    `docker run  --entrypoint bash -dit --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /nas:/nas --name sx1 <image_A>`


### 更改镜像 image_A 的入口，做成新镜像 image_B，并创建容器
* 查看有的镜像名：`docker images`，找到 <image_A>
* 更改现有镜像的入口（a 变为 b）

    ```bash
    # 从镜像a创建一个容器
    docker create --name temp <image_A>

    # 启动该容器  
    docker start temp

    # 修改entrypoint为bash
    docker commit --change "ENTRYPOINT [\"bash\"]" temp <image_B>

    # 其他修改，例如注册用户 ma-user

    # 删除临时容器          sx:torch1.11_cuda11.6
    docker rm temp
    ```
* 从更改得到的镜像 image_B 构建容器 sx1   
    * `docker run -dit --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /nas:/nas --name sx1 <image_B>`
* Debug
    * 有时会遇见容器内 `nvidia-smi` 正常，但是 `torch.cuda.is_available()` 找不到 GPU 的问题，创建容器时加上 `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility` 即可
        * 原因：在默认情况下，Docker 容器内不能直接访问宿主机的 GPU，需要配置：`--gpus all`，否则 nvidia-smi 无打印。但是默认情况下只有图形和显示相关的功能可用，这种情况可能发生在更新了宿主驱动之后。所以需要加上 `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility`，否则宿主的 nvidia driver 在容器内是仅作为 utility 存在的。其发挥作用的主要是 compute 选项，这时 driver 将对容器提供计算支持（就是cuda支持）
    * 有时容器内会提示内存不足，可以再加一个选项 `--shm-size 16G`（或更大），设置共享内存的大小，用于进程间通信
        * 上面的 `--ulimit memlock` 用于设置每个进程可以锁定的最大内存量，防止内存被swap出去
        * `--shm-size` 设置容器内共享内存的总大小，用于进程间通信

* ***最终命令：`docker run -dit --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 16G -v /nas:/nas -v /mnt:/mnt -p 6003:6003 --name sx_test images:torch1.11_cuda11.6`***，其中 `-p 6003:6003` 为了让主机和 docker 内部的 6003 联通
    * 最后进入容器：`docker start -i sx_test`
    * 可以用如下的代码测试，或用 [torch分布式通信测试代码](https://github.com/sxontheway/Keep-Learning/blob/master/Research/distributed_training.md#torch-%E6%9C%80%E5%B0%8F%E5%88%86%E5%B8%83%E5%BC%8F%E6%B5%8B%E8%AF%95%E4%BB%A3%E7%A0%81)
      ```python
        # host 的 docker，ip '10.90.91.54'
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('0.0.0.0', 6003))
        s.listen(1)
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        conn.close()
        
        # remote 的 docker 中 
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(('10.90.91.54', 6003))
            print("Connection successful")
            s.close()
        except Exception as e:
            print(f"Connection failed: {e}")
      ```



### 进阶例子：为已有镜像加一个新用户 ma-user，并做成新镜像
查看本地有哪些 `/bin/bash` 用户：`grep /etc/passwd -e '/bin/bash'`，发现没有 ma-user

为此，我要把如下命令加到一个已有的镜像 sx:torch1.11_cuda11.6 中，生成一个新镜像 sx_cloud:torch1.11_cuda11.6
```bash
groupadd ma-group -g 1000 && \
useradd -d /home/ma-user -m -u 1000 -g 1000 -s /bin/bash ma-user && \
chmod 770 /home/ma-user && \
usermod -a -G root ma-user && \
echo "ma-user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
```

实现这个的过程如下：
```bash
# 基于该镜像创建容器,并进入容器（如果报错，试试去掉 /bin/bash）
docker run -it sx:torch1.11_cuda11.6 /bin/bash

# 在容器内执行你要添加的命令
groupadd ma-group -g 1000 
useradd -d /home/ma-user -m -u 1000 -g 1000 -s /bin/bash ma-user
chmod 770 /home/ma-user
usermod -a -G root ma-user
echo "ma-user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 退出，并生成新镜像
exit
docker commit [container_id] sx_cloud:torch1.11_cuda11.6
```

### （不 work 的尝试）
系统重装了，但是保留了 `var/lib/docker/overlay2` 文件夹，尝试恢复镜像
```bash
systemctl stop docker
cp -r /mnt/sda7/backup/overlay2 /var/lib/docker/
sudo systemctl start docker
docker images
```
