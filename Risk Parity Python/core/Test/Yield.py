import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from core.riskfunction import RiskConcentrationFunction
from core.sco import CoordinateDescentOptimizer
from core.rpp import RiskParityPortfolio

dataset = pd.read_excel(
        "/Users/lizongyun/Desktop/中信建投/投研/Test.xls", sheet_name=0, header=0
    )
dataset['Time'] = pd.to_datetime(dataset['Time'], format='%Y/%m/%d')
dataset.set_index(['Time'], inplace=True)

"""
    Initiate: Set start date, end date, seasonal/annual(backlength),
    frequency for covariance, frequency for portfolio adj
"""

start = input("Please enter startdate in the form Year/Month/Day:\n")
start = pd.to_datetime(start, format='%Y/%m/%d')


end = input("Please enter enddate in the form Year/Month/Day:\n")
end = pd.to_datetime(end, format='%Y/%m/%d')


backlen = int(input("Please enter the back period for covariance:\n"
                            "unit is in months"))
backlen = relativedelta(months=backlen)


voltype = input("Please enter the type of volatility(weekly/monthly):\n")
if voltype == "weekly":
    length = 5
elif voltype == "monthly":
    length = 20


frequency = int(input("Please enter the frequency of portfolio adjustment:\n"
                          "unit is in months"))
frequency = relativedelta(months=frequency)

curtime = start

yieldfreq = int(input("Please enter the frequency of yield calculation:\n"
                          "unit is in months"))
yieldfreq = relativedelta(months=yieldfreq)

aa = np.array([])
yieldframe = np.zeros(len(dataset.columns))
#假设test为权重 权重的问题
test = np.ones(len(dataset.columns)) / 5
yieldsaver = np.array([])
count = 0
newweights = np.ones(len(dataset.columns)) / 5
np.put(newweights, 1, 0.6)
np.put(newweights, 0, 0.1)
np.put(newweights, 2, 0.1)
np.put(newweights, 3, 0.1)
np.put(newweights, 4, 0.1)

while curtime <= end:
    period = dataset.truncate(before=curtime - yieldfreq, after=curtime - datetime.timedelta(days=1))
    #算出每项资产的月总收益率
    yieldmonth = period.product()
    #把每个月的总收益率叠加
    yieldframe = np.vstack((yieldframe, yieldmonth))
    #每项资产当月的收益 = 该资产权重/价值 x 该资产当月的月收益率, 再把每项资产当月的收益加起来为当月组合总收益率
    result = np.sum(np.multiply(test, yieldmonth))
    yieldsaver = np.append(yieldsaver, result)
    #更新每项资产当月结算后的价值百分比
    if count % 3 == 0 and count != 0:
        test = np.multiply(newweights, result)
    else:
        test = np.multiply(test, yieldmonth)
    #更新时间到新一个月
    curtime += yieldfreq
    count += 1

yieldframe = np.delete(yieldframe, 0, 0)
print(yieldframe)
print(yieldsaver)


