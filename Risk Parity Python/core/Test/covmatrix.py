import pandas as pd
import numpy as np
import datetime
from _datetime import timedelta


if __name__ == "__main__":

    """
    Dataset从Excel中提取
    """
    dataset = pd.read_excel(
        "/Users/lizongyun/Desktop/中信建投/投研/工作簿1.xls", sheet_name=0, header=0
    )

    """
    1. 把Dataset中的Time Column格式转化为datetime格式
    2. 把Time列设置为index
    """
    dataset['Time'] = pd.to_datetime(dataset['Time'], format='%Y%m%d')
    dataset.set_index(['Time'], inplace=True)

    """
    ============================================================
    """

    """
    设置Truncate时间节点，begin每个loop加三个月或一年，end设置为T0
    """
    begin = datetime.datetime(2021, 6, 3)
    end = datetime.datetime(2021, 6, 6)

    """
    Truncate Dataset，此步骤之后可能需要清洗这个片段
    """
    cov = dataset.truncate(before=begin, after=end)
    print(cov)

    """
    return这个片段的协方差矩阵
    """
    cov_matrix = pd.DataFrame.cov(cov)
    print(cov_matrix)

    """
    
    """












