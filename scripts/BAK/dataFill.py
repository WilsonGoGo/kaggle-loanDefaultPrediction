import os
import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import seaborn as sns
plt.style.use('ggplot')
# %matplotlib inline
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False




### 读取数据 ###
# 设置数据路径
dataPath = r"E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\dataset"
# 读取数据
train = pd.read_csv(os.path.join(dataPath, 'train.csv'), index_col='id')    # 指定id列为行标签
test= pd.read_csv(os.path.join(dataPath, 'testA.csv'), index_col='id')      # 指定id列为行标签
# isDefault列是结果而非特征，pop掉
target = train.pop('isDefault') #目标值设置
test = test[train.columns]
ans = test.pop('isDefault')
### ####### ###



### 数值型数据分类 ###
#数值类型: numerical_feature
numerical_feature = list(train.select_dtypes(exclude=['object']).columns)
#对象类型: category_feature: ['grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine']
category_feature = list(filter(lambda x: x not in numerical_feature,list(train.columns)))
#连续型数值类型
serial_feature = []
#离散型数值类型
discrete_feature = []
# 遍历判断数值型变量是连续还是离散（以变化类型是否超过15为界限）
for feature in numerical_feature:
    temp = train[feature].nunique()
    if temp <= 15:
        discrete_feature.append(feature)
    else:
        serial_feature.append(feature)
### ############# ###


### 填充空值 ###
# 数值型用中位数，对象型用众数)
for col in numerical_feature:
    train[col].fillna(train[col].median(), inplace = True )
    test[col].fillna(train[col].median(), inplace = True)
for col in category_feature:
    train[col].fillna(train[col].mode(), inplace = True)
    test[col].fillna(train[col].mode(), inplace = True)
# 不知道为什么这一列填充总是失败，因此改用pad模式后正常
# （train['employmentLength'].mode()值返回正常）
train['employmentLength'].fillna(method='pad', inplace = True)
test['employmentLength'].fillna(method='pad', inplace = True)    
### ####### ###



### 储存数据 ###
train.to_csv('train_filledUp.csv',index=False)
target.to_csv('target_filledUp.csv',index=False)
test.to_csv('test_filledUp.csv',index=False)
ans.to_csv('ans.csv')
