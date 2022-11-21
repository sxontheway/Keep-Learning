# git 使用
> https://www.ruanyifeng.com/blog/2014/06/git_remote.html

## 一般流程
* push之前：需要 checkout，也就是对于每个发生 change 的文件决定 discard changes 还是 stage changes；当所有需要的 change 都 staged 之后，commit with message。接下来就可以和远端 sych 了
* push 和 pull：pull 的 change 和自己的 push 发生在一个文件时，会产生冲突，需要手动解决每个地方的 conflict

## 将本地和仓库同步
### 方法一：从网页创建仓库，再克隆到本地
> https://jimmylab.wordpress.com/gp1015/git-github/vscode-github/ 
1. 先在网页版建一个仓库
1. 在本地某个文件夹下运行 `git clone https://github.com/xxx/xxxx.git`
1. 进入这个 clone 的文件夹：`git config user.name xxx`, `git config user.email xxx@hotmail.com`
1. 然后就可以在本地修改，commit 后再 update，就同步了

### 方法二：将本地仓库同步到远程仓库
> [如何将本地的项目一键同步到GitHub](https://wdblink.github.io/2019/03/24/%E5%A6%82%E4%BD%95%E5%B0%86%E6%9C%AC%E5%9C%B0%E7%9A%84%E9%A1%B9%E7%9B%AE%E4%B8%80%E9%94%AE%E5%90%8C%E6%AD%A5%E5%88%B0GitHub/)

已经有本地的工程文件，同步到 git 的一般步骤是这样的：
1. 将目录变成 git 可以管理的仓库：在本地工程目录下执行 `git init`，`git add .`（建立git并开始跟踪文件） 
2. 建立和连接远程仓库。如果使用github，那么就是新建 Repository，假设想叫做 myProject。这时候，github 告诉你，可以通过一下两者之一:  
    * `git remote add origin git@github.com:sxontheway/myProject.git`，  
    * `git remote add origin https://github.com/sxontheway/myProject.git`  
    
    add 之后，可输入 `git remote` 验证
3. Push 代码：`git push origin <分支名>`

#### Debug
* VSCode 用 git 报错类似：`Failed to connect to 127.0.0.1 port 10808: Connection refused`，是由于设置了全局代理  
查询是否全局代理：`git config --global http.proxy `  
* 如果出现错误 `failed to push som refs to……`，则先把远程服务器github上面的文件拉先来，再push 上去：`git pull origin <branch>`


## git 命令
* `git status`：查看状态，可有看到哪些 change 还没有 staged
* `git config`：打印 config
    * `git config --local  --list`
    * `git config --global  --list`

* `git remote` 相关
    * `origin` 是什么？   
        > https://www.zhihu.com/question/27712995    
        
        origin是默认情况下的远端库的名字，可以理解为远程仓库在本地的标签或者别名。origin 本身只是一个标签，其实可以改成任何其他的。下文的 `<远端库>` 也即 `origin`

    * `git remote`：不带任何选项的时候，列出所有远端库的名字，例如一般会返回 origin
    * `git remote -v`：使用-v选项，可以看远端库的网址
    * `git remote show <远端库>`：查看远端库详细信息，例如 `git remote show origin`
    * `git remote add <远端库> <网址>`：添加远端库，例如 `git remote add origin https://github.com/sxontheway/<repo_name>.git`
    * `git remote rm <远端库>`：删除远端库

* `git clone`  
    克隆版本库的时候，所使用的远程主机自动被Git命名为 origin，如果想用其他的主机名，需要用`git clone`命令的`-o`选项指定：`git clone -o aaa https://github.com/jquery/jquery.git`

