# 说明
介于A股的特殊性，上市公司股东户数一般和股价有负相关关系。  
本项目的目的就是爬取A股上市公司股东人数。  
详细注释见三个.py文件。

<br>

# 步骤
* 运行 crawler.py，生成一个xxxxxxxx_xxxxxx.txt文件 (格式为时间：yymmdd_hhmmss.txt)
* 从i问财下载历史股东户数的excel (optional)
* 运行transformer，将excel转化为一个data_from_iwencai.txt文件
* 复制粘贴将两个.txt合并，命名为merged.txt，运行change_time.py，这是将数据中不开市的日期（例如四个季报日：0331, 0630, 0930, 1231) 改为最近的交易日，否则在通达信中显示不出来
* 通达信导入

<br>

# 总结
## 正则表达式
* 正则匹配中的关键字： https://blog.csdn.net/liao392781/article/details/80495411  
* 什么是原始字符串 `r'sting' `？  
    简而言之，就是方式转意字符`\`泛滥的情况，但除了正则匹配外，不建议用。  
    见 https://www.jianshu.com/p/4e883cba49ec
```
# 从类似 "共13页\n\t\t\t\t\t\t\t\t" 这种字符串中提取13
    string = "共13页\n\t\t\t\t\t\t\t\t"
    total_page = 0
    for i in re.findall(r'\d', string):
        total_page = total_page*10 + int(i)
```

## Pandas 和 BeautifulSoup
> https://zhuanlan.zhihu.com/p/51968879  
> [10行代码爬取全国所有A股/港股/新三板上市公司信息](https://mp.weixin.qq.com/s?__biz=MjM5NTY1MjY0MQ==&mid=2650743597&idx=1&sn=147a38540b1269bd08b821a3f64a57b6&chksm=befeb66389893f75950fa7f2f255329cd42d0f76454bd5300e86bbf77bee3fa64d59e21f9000&mpshare=1&scene=1&srcid=#rd)  

* 看网页信息：将鼠标移到所需要查看的地方 -> 右键 -> 查看元素
* 用Pandas可十分方便地爬取Table类型的表格，格式大致如下：
    ```
    <table class="..." id="...">
        <thead>
        <tr>
        <th>...</th>
        </tr>
        </thead>
        <tbody>
            <tr>
                <td>...</td>
            </tr>
            <tr>...</tr>
            <tr>...</tr>
            <tr>...</tr>
            <tr>...</tr>
            ...
            <tr>...</tr>
            <tr>...</tr>
            <tr>...</tr>
            <tr>...</tr>        
        </tbody>
    </table>
    ```
    关键的代码：    
    ```
    data = pd.read_html(url)[0]
    data_list = np.array(data.values.tolist()) 
    ```

* 实现其他的功能，例如翻页等，需要用BeautifulSoup去解析页面的html
    ```
    html = urllib.request.urlopen(url = url) 
    soup = BeautifulSoup(html, features = "lxml")  
    filter_page = soup.find_all("span", attrs={'class':'total'})  # 筛选标签为<span>，属性为'total'的字段
    string = filter_page[0].text    # 转化为字符串
    ```