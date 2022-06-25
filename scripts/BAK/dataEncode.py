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
dataPath = r"E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\dataset\\processed\\filledUp"
# 读取数据
train = pd.read_csv(os.path.join(dataPath, 'train_filledUp.csv')) 
test= pd.read_csv(os.path.join(dataPath, 'test_filledUp.csv')) 
target = pd.read_csv(os.path.join(dataPath, 'target_filledUp.csv')) 
### ####### ###


### 对象型数据编码 ###
# ['grade', 'subGrade']类型： 是简单的字符、字符串代码
# 考虑对这两列手动编码
a2z = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
a2z_code = np.arange(1,27)
a2z_mapping = dict(zip(a2z, a2z_code))

for data in [train,test]:
    data.loc[:,['grade','subGrade']] = data.loc[:,['grade','subGrade']].applymap(lambda g:g.replace(g[0], str(a2z.index(g[0])+1))).astype('int')

# ['employmentLength']类型： 是字符串，种类多样："<1 year", "n years", "10+ years" 共12种
# 由于种类固定，直接提取其中的数字
train['employmentLength'] = train['employmentLength'].replace({'< 1 year':'1 year','10+ years':'10 years'}).str.split(' ',expand=True)[0].astype('int64')
test['employmentLength'] = test['employmentLength'].replace({'< 1 year':'1 year','10+ years':'10 years'}).str.split(' ',expand=True)[0].astype('int64')

# ['issueDate', 'earliesCreditLine']类型： 都是年/月/日类型字符串
# 考虑将两者合并，只记录差值，存放于['CreditLine']列中
train_earliesCreditLine_year = train['earliesCreditLine'].apply(lambda x:x[-4:]).astype('int64')
test_earliesCreditLine_year = test['earliesCreditLine'].apply(lambda x:x[-4:]).astype('int64')

train_issueDate_year = train['issueDate'].apply(lambda x:x[:4]).astype('int64')
test_issueDate_year = test['issueDate'].apply(lambda x:x[:4]).astype('int64')

train['CreditLine'] = train_issueDate_year - train_earliesCreditLine_year
test['CreditLine'] = test_issueDate_year - test_earliesCreditLine_year

train = train.drop(['earliesCreditLine','issueDate'],axis=1)
test = test.drop(['earliesCreditLine','issueDate'],axis=1)
### ############# ###


### 储存数据 ###
train.to_csv('train_encoded.csv',index=False)
target.to_csv('target_encoded.csv',index=False)
test.to_csv('test_encoded.csv',index=False)

