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


### 面向对象 ###
class dataHandler :
    """
    由于原数据集中的测试集没有答案，
    因此实际操作中我将原训练集train.csv中的后1/4数据抽出用作训练集
    """
    data = pd.DataFrame()   # 存放读取的原始数据
    numerical_feature = []  # 存放数字类型特征
    category_feature = []   # 存放对象类型特征
    serial_feature = []     # 存放连续的数字类型特征
    discrete_feature = []   # 存放离散的数字类型特征

    def __init__(self, dataPath) :  
        """ 
        dataPath是train.csv的绝对路径
        """
        # 读入数据
        self.data = pd.read_csv(dataPath, index_col = 'id')
        ### 数值型数据分类 ###
        #数值类型: numerical_feature
        self.numerical_feature = list(self.data.select_dtypes(exclude=['object']).columns)
        #对象类型: category_feature: ['grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine']
        self.category_feature = list(filter(lambda x: x not in self.numerical_feature,list(self.data.columns)))
        # 遍历判断数值型变量是连续还是离散（以变化类型是否超过15为界限）
        for feature in self.numerical_feature:
            temp = self.data[feature].nunique()
            if temp <= 15:
                self.discrete_feature.append(feature)
            else:
                self.serial_feature.append(feature)

    ### 填充空值 ###
    def dataFillup(self) :
        # 数值型用中位数填充
        for col in self.numerical_feature:
            self.data[col].fillna(self.data[col].median(), inplace = True )
        # 对象型用众数填充
        for col in self.category_feature:
            self.data[col].fillna(self.data[col].mode(), inplace = True)
        # ['employmentLength']项特殊对待，用上一项进行填充
        # 不知道为什么这一列填充总是失败，因此改用pad模式后正常
        # （train['employmentLength'].mode()值返回正常）
        self.data['employmentLength'].fillna(method='pad', inplace = True)
  
    ### 对象型数据编码 ###
    def dataEncode(self) :    
        ### ['grade', 'subGrade']类型 ### 
        # 这两列是简单的字符、字符串代码，考虑对这两列手动编码
        a2z = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        a2z_code = np.arange(1,27)
        a2z_mapping = dict(zip(a2z, a2z_code))
        # 遍历对这两个特征进行替换
        self.data.loc[:,['grade','subGrade']] = self.data.loc[:,['grade','subGrade']].applymap(lambda g:g.replace(g[0], str(a2z.index(g[0])+1))).astype('int')

        ### ['employmentLength']类型 ### 
        # 该类型是字符串，种类多样："<1 year", "n years", "10+ years" 共12种
        # 由于种类固定，直接提取其中的数字
        self.data['employmentLength'] = self.data['employmentLength'].replace({'< 1 year':'1 year','10+ years':'10 years'}).str.split(' ',expand=True)[0].astype('int64')
        
        ### ['issueDate', 'earliesCreditLine']类型 ### 
        # 该类型均为年/月/日类型字符串
        # 考虑将两者合并，只记录差值，存放于['CreditLine']列中
        data_earliesCreditLine_year = self.data['earliesCreditLine'].apply(lambda x:x[-4:]).astype('int64')
        data_issueDate_year = self.data['issueDate'].apply(lambda x:x[:4]).astype('int64')
        # 将二者做差
        self.data['CreditLine'] = data_issueDate_year - data_earliesCreditLine_year
        # 丢弃原项
        self.data = self.data.drop(['earliesCreditLine','issueDate'], axis=1)

    def dataSave(self, percentage) :
        """
        percentage (float) : 是分割得到的新训练集的占比
        """
        # 按比例切分
        trainSize = round(len(self.data.index) * percentage)    # 需要round函数保证输出为整数
        train = self.data[0:trainSize]
        test = self.data[trainSize:]
        # 提取各自结果项
        target = train.pop('isDefault')
        ans = test.pop('isDefault')
        # 存储
        train.to_csv('train.csv',index=False)
        test.to_csv('test.csv',index=False)
        target.to_csv('target.csv',index=False)
        ans.to_csv('ans.csv',index=False)


if __name__ == '__main__' :
    # 读入数据
    dataPath = r"E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\dataset\\raw"
    handler = dataHandler(os.path.join(dataPath, "train.csv"))
    # 处理数据
    handler.dataFillup()    # 填充数据
    handler.dataEncode()    # 编码数据
    # 保存处理结果
    handler.dataSave(0.75)

