import os 
import pandas as pd
from utils.train import trainer

if __name__ == "__main__" :
    dataPath = r"E:\\个人文件归档\\课程文件归档\\北大课程\\下学期资料\\数据挖掘\\大作业\\天池-贷款\\output\\experiment3\\data"
    train = pd.read_csv(os.path.join(dataPath, "train.csv"))
    test = pd.read_csv(os.path.join(dataPath, "test.csv"))
    target = pd.read_csv(os.path.join(dataPath, "target.csv"))
    ans = pd.read_csv(os.path.join(dataPath, "ans.csv"))


    # 初始化类，传入数据
    model = trainer(train, test, target, ans)
    # 设置模型：用默认参数
    model.setModel(num_leaves=30, learning_rate=0.02,n_estimators=2000)
    # 验证结果
    model.localAuth()
    model.testAuth()
