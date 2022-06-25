import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def getNcomponents(train, columns) :
    train_col = train.loc[:,columns]
    ### 可视化寻找n_components最优取值 ###
    pca = PCA().fit(train_col)  # 默认n_components为特征数目4
    pca_info = pca.explained_variance_ratio_
    # print("每个特征在原始数据信息占比：\n", pca_info)
    pca_info_sum = np.cumsum(pca_info)
    # print("前i个特征总共在原始数据信息占比：\n", pca_info_sum)

    plt.plot(range(1,len(columns)+1), pca_info_sum)  # [1, 2, 3, 4]表示选1个特征、2个特征...
    plt.xticks(range(1,len(columns)+1))  # 限制坐标长度
    plt.xlabel('The number of features after dimension')
    plt.ylabel('The sum of explained_variance_ratio_')
    plt.savefig("pca")
    plt.show()



if __name__ == "__main__" :
    ### 读取数据 ###
    train = pd.read_csv("E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\output\\experiment1\\data\\train.csv")
    test = pd.read_csv("E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\output\\experiment1\\data\\test.csv")

    ### 手动降维 ###
    # 删去有效信息过少的特征
    useless_col = ["policyCode", "annualIncome", "applicationType", "n11", "n12", "n13"]
    train.drop(useless_col,axis=1,inplace=True) # axis=0表示行；axis=1表示列
    test.drop(useless_col,axis=1,inplace=True) # axis=0表示行；axis=1表示列
    # 删去可被替换的特征
    replaceable_col = ["grade"]
    train.drop(replaceable_col,axis=1,inplace=True)
    test.drop(replaceable_col,axis=1,inplace=True)


    ### pca降维 ###
    col_pca_1 = ["openAcc", "totalAcc", "n1", "n2", "n3", "n4", "n7", "n9"]
    col_pca_2 = ["loanAmnt", "installment"]
    col_pca_3 = ["interestRate", "subGrade"]
    train_col_1 = train.loc[:,col_pca_1]
    train_col_2 = train.loc[:,col_pca_2]
    train_col_3 = train.loc[:,col_pca_3]
    test_col_1 = test.loc[:,col_pca_1]
    test_col_2 = test.loc[:,col_pca_2]
    test_col_3 = test.loc[:,col_pca_3]

    ### 针对第一组求最佳n_components参数 ###
    getNcomponents(train, col_pca_1)

    ### pca降维：第一组 ###
    pca_1 =  PCA(n_components=3)    # 设置pca参数
    # 做归一化等操作后直接使用pca降维（即SVD分解）
    train_col_1 = pca_1.fit_transform(train_col_1)
    test_col_1 = pca_1.transform(test_col_1)
    # 生成存放新特征用的DataFrame数据
    train_pca_1 = pd.DataFrame(train_col_1,columns=['pca1-1','pca1-2','pca1-3'])
    test_pca_1 = pd.DataFrame(test_col_1,columns=['pca1-1','pca1-2','pca1-3'])
    # 删去原有特征
    train.drop(col_pca_1,axis=1,inplace=True)
    test.drop(col_pca_1,axis=1,inplace=True)
    # 添加入新特征至原DataFrame数据
    train = pd.concat([train,train_pca_1],axis=1)
    test = pd.concat([test,test_pca_1],axis=1)


    ### pca降维：第二组 ###
    pca_2 =  PCA(n_components=1)

    train_col_2 = pca_2.fit_transform(train_col_2)
    test_col_2 = pca_2.transform(test_col_2)

    train_pca_2 = pd.DataFrame(train_col_2,columns=['pca2'])
    test_pca_2 = pd.DataFrame(test_col_2,columns=['pca2'])

    train.drop(col_pca_2,axis=1,inplace=True)
    test.drop(col_pca_2,axis=1,inplace=True)

    train = pd.concat([train,train_pca_2],axis=1)
    test = pd.concat([test,test_pca_2],axis=1)

    ### pca降维：第三组 ###
    pca_3 =  PCA(n_components=1)

    train_col_3 = pca_3.fit_transform(train_col_3)
    test_col_3 = pca_3.transform(test_col_3)

    train_pca_3 = pd.DataFrame(train_col_3,columns=['pca3'])
    test_pca_3 = pd.DataFrame(test_col_3,columns=['pca3'])

    train.drop(col_pca_3,axis=1,inplace=True)
    test.drop(col_pca_3,axis=1,inplace=True)

    train = pd.concat([train,train_pca_3],axis=1)
    test = pd.concat([test,test_pca_3],axis=1)

    ### 保存新数据 ###
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)