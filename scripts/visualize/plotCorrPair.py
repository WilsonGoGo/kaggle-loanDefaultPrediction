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



def plotCorrPair(data, feature_x, feature_y) :
    plt.scatter(data[feature_x], data[feature_y])
    plt.xlabel(feature_x + '-' + feature_y)
    plt.savefig(feature_x + '-' + feature_y + ".png")
    plt.clf()


if __name__ == "__main__" :
    train = pd.read_csv("E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\dataset\\processed\\encoded\\train_encoded.csv", index_col=False)
    featurePairList = [
        ["loanAmnt", "installment"],
        ["interestRate", "grade"],
        ["interestRate", "subGrade"],
        ["grade", "subGrade"],
        ["ficoRangeHigh", "ficoRangeLow"],
        ["openAcc", "totalAcc"],
        ["openAcc", "n7"],
        ["openAcc", "n9"]#,
        ["n2", "n3"],
        ["n2", "n9"],
        ["n3", "n9"],
        ["n1", "n2"],
        ["n1", "n3"],
        ["n1", "n4"],
        ["n1", "n9"]
    ]

    for pair in featurePairList :
        plotCorrPair(train, pair[0], pair[1])