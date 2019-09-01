# -*- coding: utf-8 -*-

# 本脚本用于爬取最新的股东户数信息，并转化为通达信外部序列数据格式
# 一点仓位(http://www.yidiancangwei.com)上，股东户数信息每15日更新一次

import pandas as pd
from bs4 import  BeautifulSoup
import re
import urllib.request
import numpy as np
import datetime


# 从HTML中，提取切换时间段那个按钮的URL所需参数
prefix = "window.location.href='"            
attribute = "onclick"
init_url = "http://www.yidiancangwei.com"   # 必须要加上http://


# 提取切换时间段的那些按钮的URL
'''
从如下示例 html 中提取相对URL： "index.php?StartDay=2019-08-01&amp;EndDay=2019-08-15&amp;Page=1" 
    <div class="Time " data-type="1" onclick="window.location.href='index.php?StartDay=2019-08-01&amp;EndDay=2019-08-15&amp;Page=1'">
    08月01日-08月15日
    </div>
'''
def get_url(prefix, attribute, init_url):
    html = urllib.request.urlopen(url = init_url)   # 获取html
    soup = BeautifulSoup(html, features = "lxml")   # 用BeautifulSoup解析html
    filter_url = soup.find_all("div", attrs={'class':'Time'})    # 用标签和属性筛选html
    urls = []
    for i in filter_url:
        str_pos = len(prefix) 
        full_url = init_url + "/" + i.attrs[attribute][str_pos:-1]   # 截取相对URL，并和根URL合并
        urls.append(full_url)
    return urls


# 提取总页数
def total_page_num(url):
    html = urllib.request.urlopen(url = url)   # 获取html
    soup = BeautifulSoup(html, features = "lxml")   # 用BeautifulSoup解析html  
    filter_page = soup.find_all("span", attrs={'class':'total'})  # 用标签和属性筛选html
    string = filter_page[0].text    # 得到的字符串为 "共13页\n\t\t\t\t\t\t\t\t" 这种格式
    total_page = int(string.strip()[1:-1])

    '''
    # 或者用正则表达式
    total_page = 0
    for i in re.findall(r'\d', string):
        total_page = total_page*10 + int(i)
    '''
    return total_page


# 用pandas爬取表格，输出格式为一个np.array
def pandas_crawler(url):
    data = pd.read_html(url)[0]
    return np.array(data.values.tolist())   # 转成numpy array方便索引


# 将表格数据转换为通达信序列数据格式，并写入文件
'''
0代表深证，1代表上证；时间由老到新；最后一列为所要导入的数据
    0|000001|20170630|379200
    0|000001|20170929|331500
    0|000002|20160930|321500
    0|000002|20170929|208900
'''
def process_form(array):    
    filtered = np.zeros((array.shape[0], 4)).astype(np.str)
    filtered[:, 1] = [y.zfill(6) for y in array[:, 2]]      # 股票代码
    filtered[:, 2] = [int(''.join(y.split('-'))) for y in array[:, 5]]   # 日期
    filtered[:, 3] = [round(float(y)) if y != '-' else 0 for y in array[:, 3]]    # 所要导入数据
    # 判断是深市还是沪市：000,001,002,200,300,900为深证；6开头为上证
    filtered[:, 0] = [1 if y[0] == '6' else 0 for y in filtered[:, 1]]
    return filtered


# 对所爬取到的表格进行排序,并打印
def rank(data, f):
    ranked_data = sorted(data, key = lambda x: (x[1], x[2]))
    print(ranked_data)
    for i in ranked_data:
        f.write(i[0]+"|"+i[1]+'|'+i[2]+'|'+i[3]+'\n')



if __name__ == '__main__':
     # 提取切换时间段的那些按钮的URL，输出格式为一个list
    urls = get_url(prefix, attribute, init_url)    

    # 提取上述每个URL下的总页码数，输出格式为一个list
    page_nums = []
    for url in urls:
        page_nums.append(total_page_num(url))   

    # 逐页面爬取表格
    data = np.empty((0, 4))
    for item, url in enumerate(urls):
        page_num = page_nums[item]
        for i in range(page_num):
            new_url = url[:-1] + str(i+1)   # 此处得到了所有页面完整的URL列表
            data = np.append(data, process_form(pandas_crawler(new_url)), axis = 0)
            print(new_url + '爬取完成')

    # 对数据进行排序，并写入文件 
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"      
    with open(filename,"a") as f:
        rank(data, f)