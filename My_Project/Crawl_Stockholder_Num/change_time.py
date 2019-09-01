# -*- coding: utf-8 -*-
import tushare as ts
import datetime

# 全局变量
alldays = ts.trade_cal()    
alldays_list = alldays['calendarDate'].tolist()

# 用tushare包，得到最近的交易日期
def last_open_day(date):
    date = date[:4] + "-" + date[4:6]+ "-"+ date[6:8]
    date_index = alldays_list.index(date)

    index = date_index
    while( alldays['isOpen'][index] != 1 ):
        index -= 1
        
    return(''.join(alldays_list[index].split('-')))


new_data = []

with open("merged.txt","r+") as f:
    for line in f.readlines():
        new_data.append( line.replace(line[9:17], last_open_day(line[9:17])) )

with open("merged1.txt","w") as f:
    for i in new_data:
        f.write(i)
