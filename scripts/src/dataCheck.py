import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def collect_na_value(dataframe):
    return dataframe.isna().sum() / dataframe.shape[0] * 100

if __name__ == "__main__" :
    ### 读取数据 ###
    data = pd.read_csv("E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\dataset\\raw\\train.csv", index_col = 'id')
    ### 读取缺失项 ###
    null_info = collect_na_value(data)
    ### 可视化 ###
    plt.figure()
    plt.plot(null_info)
    plt.show()
