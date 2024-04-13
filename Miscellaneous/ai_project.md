# 服务器
* 青云
    * 参见 fastgpt 需求，先暂时租 2c4g 60GB https://console.qingcloud.com/ap2a
    * 申请公网 ip（付费），配置 ssh，rsa 密钥
* docker-comose http://doc.fastai.site/docs/development/docker/
    * docker 安装 oneapi，mongo，pg，fastgpt，mysql
        ```
        # 安装 Docker
        curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
        systemctl enable --now docker
        # 安装 docker-compose
        curl -L https://github.com/docker/compose/releases/download/v2.20.3/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
        # 验证安装
        docker -v
        docker-compose -v

        mkdir fastgpt
        cd fastgpt
        curl -O https://raw.githubusercontent.com/labring/FastGPT/main/files/deploy/fastgpt/docker-compose.yml
        curl -O https://raw.githubusercontent.com/labring/FastGPT/main/projects/app/data/config.json

        # 启动容器
        docker-compose up -d
        # 等待10s，OneAPI第一次总是要重启几次才能连上Mysql
        sleep 10
        # 重启一次oneapi(由于OneAPI的默认Key有点问题，不重启的话会提示找不到渠道，临时手动重启一次解决，等待作者修复)
        docker restart oneapi
        ```
    * vscode 配置端口转发 3000，3001 remote 到 localhost
        * fastgpt 端口 3000
            * root，DEFAULT_ROOT_PSW 是 1234
        * oneapi 端口 3001，root 123456 登录
            * 新建一个 kimi 渠道：sk-XXX
            * oneapi 令牌：sk-XXX

* FAQ
    * 找不到渠道 
        * OneAPI 中的渠道测试成功后，令牌名称可以随便填 比如 kimi-8k。但 fastgpt 中模型 "llmModels" 中的 "model" 必须和 OneAPI 渠道中的模型名一致。例如 moonshot-v1-8k 之类的，不能是 kimi-8k
    * OneAPI 中测试通过，但 fastGPT 中 Connection Error
        * 主要是 `OPENAI_BASE_URL=http://10.XXX.XXX.XXX:3001/v1` 必须要填对
        * 这里输入命令 `ifconfig | grep "inet " | grep -v 127.0.0.1`，找到 eth0 的地址，也即本机 host 地址（是为了让 fastgpt 的镜像能访问 oneapi）
            * oneapi 的端口映射是 `0.0.0.0:3001->3000/tcp`，把本机的 3001 映射到镜像内的 3000；所以要访问 oneapi，访问本机的 3001 即可
            * 见 https://github.com/labring/FastGPT/issues/973#issuecomment-2017411436 

* chatgpt-on-wechat
    * https://github.com/zhayujie/chatgpt-on-wechat

