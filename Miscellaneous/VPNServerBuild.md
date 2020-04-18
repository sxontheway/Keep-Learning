# 1. 用Amazon Lightsail创建实例
Lightsail是亚马逊提供的适用于个人的虚拟服务器，每月收取固定的费用，而不像EC2安装使用的资源收费。  
详见 https://www.heartnn.com/2018/05/11/deploy-shadowsocks-on-amazon-lightsail/ 
> 在不用了之后，为避免扣款，需要 1.删除实例 2.同时删除未附着实例的静态IP

<br>

# 2. 用虚拟机ssh登录Server
这一步的目的是为了方便复制第3步的命令，详见 https://www.cnblogs.com/liubin0509/p/6211909.html
* 给Server设置一个静态ip
* 下载密钥，例如`this_is_a_key.pem`
* 将复制密钥到`~/.ssh`目录
* 设置密钥权限为600，例如 `sudo chmod 600 ~/.ssh/this_is_a_key.pem`
* 加入ssh agent，例如：`ssh-add this_is_a_key`
* ssh连接，例如：`ssh ubuntu@52.199.223.188`  

<br>

# 3. 在Server上安装Shadowsocks
详见 https://www.heartnn.com/2018/05/11/deploy-shadowsocks-on-amazon-lightsail/ 
关键步骤：
```bash
# 更新apt，将内核更新到最新版
sudo apt update && sudo apt upgrade -y
sudo apt install --install-recommends linux-generic-hwe-16.04
sudo apt autoremove  # 重启后可选择删除旧内核

# 重启后执行uname -r，查看内核版本是否大于4.9。如果是则为成功，可以继续进行。

# 切换为root账户(可以先通过sudo passwd root来更改root账户的密码)
su

# 执行lsmod | grep bbr，此时结果中应该没有tcp_bbr。如果有则为成功，可以继续进行。

modprobe tcp_bbr
echo "tcp_bbr" >> /etc/modules-load.d/modules.conf
echo "net.core.default_qdisc=fq" >> /etc/sysctl.conf
echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf
sysctl -p

# 验证下面两条语句，如果上面的执行结果都包含bbr，则说明内核开启bbr成功。
sysctl net.ipv4.tcp_available_congestion_control
sysctl net.ipv4.tcp_congestion_control

# 再次验证 lsmod | grep bbr，查看是否有tcp_bbr模块。应该会有了。

# 最后使用Shadowsocks一键安装脚本，依旧是使用root账户，执行：
wget --no-check-certificate -O shadowsocks-all.sh https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-all.sh
chmod +x shadowsocks-all.sh
./shadowsocks-all.sh 2>&1 | tee shadowsocks-all.log
# 如果提示没有wget，可以先执行apt install wget。安装时选择Shadowsocks-libev版，加密方式可以选择aes-256-gcm或xchacha20-ietf-poly1305。
# 安装时最好安装simple-obfs，混淆选http还是tls可随意。(这里执行autoconf --version查询版本应该是没有问题的，所以可以正常安装。)
```

<br>


# 4. 使用Shadowsocks
详见 https://ssr.tools/386  
* Shadowsocks Windows版本下载 https://github.com/shadowsocks/shadowsocks-windows/releases/
，选PAC模式或全局模式（PAC使用GFWList列表，绕过大陆地址，但不一定稳定）
* iOS可用Outline，Android可用Shadowsocks

------
<br><br>
