## 使用git
> https://jimmylab.wordpress.com/gp1015/git-github/vscode-github/ 
* 先在网页版建一个仓库
* 在本地某个文件夹下运行 `git clone https://github.com/xxx/xxxx.git`
* 进入这个clone的文件夹：`git config user.name xxx`, `git config user.email xxx@hotmail.com`
* 然后就可以在本地修改，commit后再update，就同步了

## 将本地仓库同步到远程仓库
> [如何将本地的项目一键同步到GitHub](https://wdblink.github.io/2019/03/24/%E5%A6%82%E4%BD%95%E5%B0%86%E6%9C%AC%E5%9C%B0%E7%9A%84%E9%A1%B9%E7%9B%AE%E4%B8%80%E9%94%AE%E5%90%8C%E6%AD%A5%E5%88%B0GitHub/)
* 将本地仓库和远程仓库关联： `git remote add origin https://github.com/sxontheway/<repo_name>.git`
* push: `git push origin master`，新版 github 里面 `master` 命名为了 `main`
