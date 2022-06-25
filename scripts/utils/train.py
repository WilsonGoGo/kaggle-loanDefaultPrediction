import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import seaborn as sns
plt.style.use('ggplot')
# %matplotlib inline
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False


class trainer :
    """
    该类用于设置训练模型
    并验证训练模型的准确率
    """
    train = pd.DataFrame()      # 训练集，不包含结果['isDefault']
    test = pd.DataFrame()       # 测试集，不包含结果['isDefault']
    target = pd.DataFrame()     # 训练集的结果，仅包含原数据集的['isDefault']项
    ans = pd.DataFrame()        # 测试集的结果，仅包含原数据集的['isDefault']项
    prediction = pd.DataFrame()    # 存放测试集预测结果
    lgbr = LGBMRegressor()      # 用于存放训练模型，在setModel()函数中实例化

    def __init__(self, train, test, target, ans) :
        """
        直接传入dataframe格式数据(无['id']列)
        """
        self.train = train
        self.test = test
        self.target = target
        self.ans = ans

    def setModel(self 
            ,num_leaves=30
            ,max_depth=5
            ,learning_rate=.02
            ,n_estimators=1000
            ,subsample_for_bin=5000
            ,min_child_samples=200
            ,colsample_bytree=.2
            ,reg_alpha=.1
            ,reg_lambda=.1 
            ) :
        """
        设置训练模型参数
        """
        self.lgbr = LGBMRegressor(num_leaves=num_leaves
                        ,max_depth=max_depth
                        ,learning_rate=learning_rate
                        ,n_estimators=n_estimators
                        ,subsample_for_bin=subsample_for_bin
                        ,min_child_samples=min_child_samples
                        ,colsample_bytree=colsample_bytree
                        ,reg_alpha=reg_alpha
                        ,reg_lambda=reg_lambda
                        )
        return self.lgbr

    ### 本地验证 ###
    def localAuth(self) :
        """
        在训练集上验证
        """
        print("localAuth...")
        # 使用K-Fold验证
        kf = KFold(n_splits=10, shuffle=True, random_state=100)
        devscore = []
        for tidx, didx in kf.split(self.train.index):
            tf = self.train.iloc[tidx]
            df = self.train.iloc[didx]
            tt = self.target.iloc[tidx]
            dt = self.target.iloc[didx]
            #原版本 
            # te = TargetEncoder(cols=['subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine'])
            #第一次试验版本 
            te = TargetEncoder(cols=['subGrade', 'employmentLength', 'CreditLine'])
            # 第二次试验版本
            # te = TargetEncoder(cols=['employmentLength', 'CreditLine'])
            tf = te.fit_transform(tf, tt)
            df = te.transform(df)
            # lgbr需要提前调用self.setModel()设置好参数
            self.lgbr.fit(tf, tt)
            pre = self.lgbr.predict(df)
            fpr, tpr, thresholds = roc_curve(dt, pre)
            score = auc(fpr, tpr)
            devscore.append(score)
        # 打印结果
        print("localAuth is :")
        print(np.mean(devscore))
        # 返回结果
        return np.mean(devscore)

    ### 测试集验证 ###
    def testAuth(self) :
        """
        在测试集上验证
        """
        print("Predicting ...")
        # te = TargetEncoder(cols=['subGrade', 'employmentLength', 'CreditLine'])
        #第一次试验版本 
        te = TargetEncoder(cols=['subGrade', 'employmentLength', 'CreditLine'])
        # 第二次试验版本
        # te = TargetEncoder(cols=['employmentLength', 'CreditLine'])
        tf = te.fit_transform(self.train, self.target)
        df = te.transform(self.test)
        # 利用模型预测答案
        self.lgbr.fit(tf, self.target)
        self.prediction = self.lgbr.predict(df)
        # 存储预测结果
        print("Prediction saving ...")
        pd.Series(self.prediction, name='isDefault', index=self.test.index).reset_index().to_csv('submit.csv', index=False)
        # 利用auc计算预测指标
        print("Prediction authenticating ...")
        fpr, tpr, thresholds = roc_curve(self.ans, self.prediction)
        score = auc(fpr, tpr)
        print("Prediction auc is :")
        print(score)
        return score


