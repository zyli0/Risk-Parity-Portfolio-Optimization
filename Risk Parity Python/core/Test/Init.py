import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from core.riskfunction import RiskConcentrationFunction
from core.sco import CoordinateDescentOptimizer
from core.rpp import RiskParityPortfolio
import seaborn as sns
from collections import OrderedDict


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

yieldfreq = int(input("Please enter the frequency of yield calculation:\n"
                          "unit is in months"))
yieldfreq = relativedelta(months=yieldfreq)



"""
    Set up dataset
"""

dataset = pd.read_excel(
        "/Users/lizongyun/Desktop/中信建投/投研/Data.xls", sheet_name=1, header=0
    )
dataset['Time'] = pd.to_datetime(dataset['Time'], format='%Y/%m/%d')
dataset.set_index(['Time'], inplace=True)


curtime = start
wsaver = np.zeros(len(dataset.columns))
toversaver = np.array([])
adjsaver = np.array([])
trcsaver = np.zeros(len(dataset.columns))


def process(dataframe, length):
    if int(dataframe.shape[0]) % length == 0:
        ar = np.zeros((int((dataframe.shape[0]) / length)))
    else:
        ar = np.zeros((int((dataframe.shape[0]) / length)) + 1)

    for j in range(len(dataframe.columns)):
        parray = np.array([])
        i = 0
        while i <= dataframe.shape[0] - length:
            wyield = 1 + dataframe.iloc[i, j]
            for count in range(1, length):
                i += 1
                wyield = wyield * (1 + dataframe.iloc[i, j])
                count += 1
            parray = np.append(parray, wyield)
            i += 1

        numleft = dataframe.shape[0] - i

        """
        想想最后两种情况， 剩一个数字和0个数字
        """
        if numleft == 0:
            ar = np.column_stack((ar, parray))
        else:
            wyield = 1 + dataframe.iloc[i, j]
            while numleft > 1:
                i += 1
                wyield = wyield * (1 + dataframe.iloc[i, j])
                numleft -= 1
            parray = np.append(parray, wyield)
            ar = np.column_stack((ar, parray))

        j += 1

    ar = np.delete(ar, 0, 1)
    newframe = pd.DataFrame(ar)
    return newframe


while curtime <= end:
    if curtime == end:
        """
        计算最后一个period的收益率, 存入收益率array
        """
        wsaver = np.delete(wsaver, 0, 0)
        trcsaver = np.delete(trcsaver, 0, 0)
        toversaver = np.delete(toversaver, 0, 0)
        break
    srtime = curtime.strftime("%Y/%m/%d")
    adjsaver = np.append(adjsaver, srtime)
    period = dataset.truncate(before=curtime - backlen, after=curtime - datetime.timedelta(days=1))
    #计算每种资产改时间范围内的收益率 （1+R)(1+R).. 用的weight是
    #计算完每一种资产的收益率后，存入累计收益率array
    starperiod = process(period, length)

    cov_matrix = pd.DataFrame.cov(starperiod)
    portfolio = RiskParityPortfolio(cov_matrix)
    portfolio.design_portfolio()
    lastw = wsaver[-1]
    tover = np.sum(abs(portfolio.weights - lastw)) / 2
    wsaver = np.vstack((wsaver, portfolio.weights))
    trcsaver = np.vstack((trcsaver, portfolio.risk_contribution))
    toversaver = np.append(toversaver, tover)
    print(tover)
    curtime = curtime + frequency

print(wsaver)
print(trcsaver)
print(adjsaver)
print(toversaver)
print(np.average(toversaver))


assetnames = list(dataset.columns)
wsaverframe = pd.DataFrame(wsaver, columns=assetnames)
print(wsaverframe)
g = sns.FacetGrid(wsaverframe, col='A股', col_wrap=5)
print(g)

"""
计算收益率区域
"""

dataset = pd.read_excel(
        "/Users/lizongyun/Desktop/中信建投/投研/Data.xls", sheet_name=3, header=0
    )
dataset['Time'] = pd.to_datetime(dataset['Time'], format='%Y/%m/%d')
dataset.set_index(['Time'], inplace=True)


"""
    Initiate: Set start date, end date, seasonal/annual(backlength),
    frequency for covariance, frequency for portfolio adj
"""

start = input("Please enter startdate of yield calculation in the form Year/Month/Day:\n")
start = pd.to_datetime(start, format='%Y/%m/%d')


end = input("Please enter enddate in the form Year/Month/Day:\n")
end = pd.to_datetime(end, format='%Y/%m/%d')


frequency = int(input("Please enter the frequency of portfolio adjustment:\n"
                          "unit is in months"))
frequency = relativedelta(months=frequency)

curtime = start

yieldfreq = int(input("Please enter the frequency of yield calculation:\n"
                          "unit is in months"))
freq = yieldfreq
yieldfreq = relativedelta(months=yieldfreq)


aa = np.array([])
yieldframe = np.zeros(len(dataset.columns))
#假设test为权重 权重的问题
num = 0
curweight = wsaver[num]
yieldsaver = np.array([])
count = 1

while curtime <= end:
    period = dataset.truncate(before=curtime - yieldfreq, after=curtime - datetime.timedelta(days=1))
    #算出每项资产的月总收益率
    yieldmonth = period.product()
    #把每个月的总收益率叠加
    yieldframe = np.vstack((yieldframe, yieldmonth))
    #每项资产当月的收益 = 该资产权重/价值 x 该资产当月的月收益率, 再把每项资产当月的收益加起来为当月组合总收益率
    result = np.sum(np.multiply(curweight, yieldmonth))
    yieldsaver = np.append(yieldsaver, result)
    #更新每项资产当月结算后的价值百分比
    if curtime == end:
        break
    if count % freq == 0 and count != 0:
        num += 1
        curweight = wsaver[num]
        curweight = np.multiply(curweight, result)
    else:
        curweight = np.multiply(curweight, yieldmonth)
    #更新时间到新一个月
    curtime += yieldfreq
    count += 1

yieldframe = np.delete(yieldframe, 0, 0)
print(yieldframe)
print(yieldsaver)


"""
画图
"""

asset1 = wsaver[:, 0]
asset2 = wsaver[:, 1]
asset3 = wsaver[:, 2]
asset4 = wsaver[:, 3]
asset5 = wsaver[:, 4]
"""
asset6 = wsaver[:, 5]
asset7 = wsaver[:, 6]
asset8 = wsaver[:, 7]
asset9 = wsaver[:, 8]
asset10 = wsaver[:, 9]
asset11 = wsaver[:, 10]
asset12 = wsaver[:, 11]
asset13 = wsaver[:, 12]
asset14 = wsaver[:, 13]
asset15 = wsaver[:, 14]
asset16 = wsaver[:, 15]
asset17 = wsaver[:, 16]
asset18 = wsaver[:, 17]
asset19 = wsaver[:, 18]
asset20 = wsaver[:, 19]
asset21 = wsaver[:, 20]
asset22 = wsaver[:, 21]
asset23 = wsaver[:, 22]
asset24 = wsaver[:, 23]
asset25 = wsaver[:, 24]
asset26 = wsaver[:, 25]
asset27 = wsaver[:, 26]
asset28 = wsaver[:, 27]
"""


asset1 = asset1.tolist()
asset2 = asset2.tolist()
asset3 = asset3.tolist()
asset4 = asset4.tolist()
asset5 = asset5.tolist()
"""
asset6 = asset6.tolist()
asset7 = asset7.tolist()
asset8 = asset8.tolist()
asset9 = asset9.tolist()
asset10 = asset10.tolist()
asset11 = asset11.tolist()
asset12 = asset12.tolist()
asset13 = asset13.tolist()
asset14 = asset14.tolist()
asset15 = asset15.tolist()
asset16 = asset16.tolist()
asset17 = asset17.tolist()
asset18 = asset18.tolist()
asset19 = asset19.tolist()
asset20 = asset20.tolist()
asset21 = asset21.tolist()
asset22 = asset22.tolist()
asset23 = asset23.tolist()
asset24 = asset24.tolist()
asset25 = asset25.tolist()
asset26 = asset26.tolist()
asset27 = asset27.tolist()
asset28 = asset28.tolist()
"""


plt.rc("font", family='Songti SC')
label = list(dataset.columns.values.tolist())
print(label)

"""
28行业图
plt.stackplot(adjsaver, asset1, asset2, asset3, asset4, asset5, asset6, asset7, asset8,
              asset9, asset10, asset11, asset12, asset13, asset14, asset15, asset16,
              asset17, asset18, asset19, asset20, asset21, asset22, asset23, asset24,
              asset25, asset26, asset27, asset28, labels=label)
"""

"""
5多维度资产图
plt.stackplot(adjsaver, asset1, asset2, asset3, asset4, asset5, labels=label)
"""



"""
收益率曲线图
plt.plot(yieldsaver)
plt.show()
"""

"""
导出到Excel
"""
wframe = pd.DataFrame(wsaver)
yieldframe = pd.DataFrame(yieldsaver)
wframe.to_excel('/Users/lizongyun/Desktop/中信建投/投研/Result_Comp.xls', sheet_name='One')
yieldframe.to_excel('/Users/lizongyun/Desktop/中信建投/投研/Result_Yield.xls', sheet_name='One')

