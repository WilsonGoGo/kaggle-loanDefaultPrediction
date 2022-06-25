import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
plt.style.use('ggplot')
# %matplotlib inline
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False





def plotDistribution(data=pd.DataFrame(), label='') :   # data是datafame类型，label是列标签
    # 画图
    plt.figure(1 , figsize = (8 , 5))
    sns.distplot(train[label],bins=40)
    plt.xlabel(label)
    # 保存
    plt.savefig(label)
    plt.close()



if __name__ == "__main__" :
    train = pd.read_csv("E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\dataset\\processed\\encoded\\train_encoded.csv", index_col=False)
    for col in train.columns :
        plotDistribution(train, col)