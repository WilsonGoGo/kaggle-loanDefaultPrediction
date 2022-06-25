### 工作预期 ###
# 1.封装FeatureVisualize类，对各特征值进行可视化
# 2.填充缺失数据
# 3.尝试训练模型
# 4.实现judge功能（AUC评价指标）

### 数据读取部分 ###
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


### 读取数据 ###
# 设置数据路径
dataPath = r"E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\dataset\\processed\\encoded"
# 读取数据
train = pd.read_csv(os.path.join(dataPath, 'train_encoded.csv'))
test= pd.read_csv(os.path.join(dataPath, 'test_encoded.csv'))
target= pd.read_csv(os.path.join(dataPath, 'target_encoded.csv'))


### 模型设置部分 ###
# 模型,使用LGBMRegressor模型，在本地使用KFold进行交叉验证，由于计算资源有限，因此未使用超参数网格搜索。
def makelgb():
    lgbr = LGBMRegressor(num_leaves=30
                        ,max_depth=5
                        ,learning_rate=.02
                        ,n_estimators=1000
                        ,subsample_for_bin=5000
                        ,min_child_samples=200
                        ,colsample_bytree=.2
                        ,reg_alpha=.1
                        ,reg_lambda=.1
                        )
    return lgbr

### 训练验证部分 ###
# 本地验证
kf = KFold(n_splits=10, shuffle=True, random_state=100)
devscore = []
for tidx, didx in kf.split(train.index):
    tf = train.iloc[tidx]
    df = train.iloc[didx]
    tt = target.iloc[tidx]
    dt = target.iloc[didx]
    # te = TargetEncoder(cols=['subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine'])
    te = TargetEncoder(cols=['subGrade', 'employmentLength', 'CreditLine'])
    tf = te.fit_transform(tf, tt)
    df = te.transform(df)
    lgbr = makelgb()
    lgbr.fit(tf, tt)
    pre = lgbr.predict(df)
    fpr, tpr, thresholds = roc_curve(dt, pre)
    score = auc(fpr, tpr)
    devscore.append(score)
print(np.mean(devscore))


# 在整个train集上重新训练，预测test，输出结果
lgbr = makelgb()
# te = TargetEncoder(cols=['subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine'])
te = TargetEncoder(cols=['subGrade', 'employmentLength', 'CreditLine'])
tf = te.fit_transform(train, target)
df = te.transform(test)
lgbr.fit(tf, target)
pre = lgbr.predict(df)
pd.Series(pre, name='isDefault', index=test.index).reset_index().to_csv('submit.csv', index=False)