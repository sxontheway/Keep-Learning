# -*- coding: utf-8 -*-

# 本脚本用于清洗股东户数信息的历史数据，转化为通达信外部序列数据格式
# 历史股东户数信息可从i问财(https://www.iwencai.com/)上下载，为excel表格

import pandas as pd
import numpy as np

# 用pandas读取excel表格
form = pd.read_html('20190901.xls')[0].values.tolist()
form = np.array(form[1:])


# 将表格数据转换为通达信序列数据格式，并写入文件
'''
0代表深证，1代表上证；时间由老到新；最后一列为所要导入的数据
    0|000001|20170630|379200
    0|000001|20170929|331500
    0|000002|20160930|321500
    0|000002|20170929|208900
'''
time_list = ["20190331", "20181231", "20180930"]
data = np.empty((0, 4))

for item, time in enumerate(time_list):
    for i in form:
        stock_num = i[0].split('.')[0]
        stock_market = 1 if stock_num[0] == '6' else 0
        if i[item+4] != '--':
            data = np.append(data, np.array([[stock_market, stock_num, time, i[item+4]]]), axis = 0)


# 对所爬取到的表格进行排序,并打印
ranked_data = sorted(data, key = lambda x: (x[1], x[2]))
with open("data_from_iwencai.txt","a") as f:
    for i in ranked_data:
        f.write(i[0]+"|"+i[1]+'|'+i[2]+'|'+i[3]+'\n') 
