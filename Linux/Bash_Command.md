# Bash语法
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

# Bash命令
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

<br>

# Bash 上手
> https://www.cnblogs.com/balaamwe/archive/2012/03/15/2397998.html

* 符号
    * 其中  `$() ` 和` `` `(反引号)作用相同，是把括号内命令的输出再作为命令执行
    * `${}`括号中放的是变量，`$()`中放的是命令
* 用法
    * `2>&1 | tee XXX.log`
        ```bash
        # 其中 2>&1 means "send any error messages (aka 'stderr') to the same output as any informational messages (aka 'stdout")." 
        # | tee XXX.log means "whatever output there is should also be sent to the file XXX.log"
        ```
    * 
