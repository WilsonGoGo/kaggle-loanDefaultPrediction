import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve


### 本地验证寻找最合适的参数 ###
if __name__ == "__main__" : 
    ### 读取数据 ###
    dataPath = r"E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\output\\experiment4\\data"
    train = pd.read_csv(os.path.join(dataPath, "train.csv"), index_col=False)
    test = pd.read_csv(os.path.join(dataPath, "test.csv"), index_col=False)
    target = pd.read_csv(os.path.join(dataPath, "target.csv"), index_col=False)
    ans = pd.read_csv(os.path.join(dataPath, "ans.csv"), index_col=False)
    ### 设置预选参数组合 ###
    # parameters = {
    #     "num_leaves" : [28,30,32,34]
    #     ,"max_depth" : [3,4,5]
    #     ,"learning_rate" : [0.01,0.02,0.05,0.1,0.2]
    #     ,"n_estimators" : [500,1000,1500,2000]
    #     ,"subsample_for_bin" : [3000,5000,7000]
    #     ,"min_child_samples" : [100,200,400]
    #     }
    parameters = {
        "num_leaves" : [28,30,32]
        ,"learning_rate" : [0.01,0.02,0.05]
        ,"n_estimators" : [1500,2000,3000]
        }

    ### 初始化模型 ###
    gbm = lgb.LGBMClassifier(num_leaves=30
                            ,max_depth=5
                            ,learning_rate=.02
                            ,n_estimators=1000
                            ,subsample_for_bin=5000
                            ,min_child_samples=200
                            ,colsample_bytree=.2
                            ,reg_alpha=.1
                            ,reg_lambda=.1)
    ### kFold在原集合上测试 ###                            
    kf = KFold(n_splits=10, shuffle=True, random_state=100)
    ### GridSearchCV设置超参数网格搜索 ###
    gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
    # 直接在训练集上找最优参数
    gsearch.fit(train, target.values.ravel())

    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))