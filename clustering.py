

import pandas as pd
from utils import readDatabyState
from utils import splitDataByYear,aggrDataByYear,makeClusterData,myPCA,myStandard,myplot
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



#读取全部州的数据
raw_state_data = {}
stateList = ['Alabama','Alaska','Arizona','Arkansas']
for state in stateList:
    raw_state_data[state]  = readDatabyState(state)


#把数据按照时期聚合
year_dict = {
    'period1': [1963, 1964, 1965],
    'period2': [1966, 1967, 1968],
    'period3': [1969, 1970, 1971, 1972]
}
split_state_all_period = splitDataByYear(raw_state_data,year_dict)
aggr_state_all_period = aggrDataByYear(split_state_all_period)



# 进行机器学习聚类操作
ml_data = makeClusterData(aggr_state_all_period,year_dict)
for key_period,value_data in ml_data.items():
    subdata = ml_data[key_period]

    # 保存州的索引,最后添加到每个聚类的圆点旁边
    label_states = subdata.index

    subdata = myStandard(subdata)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(subdata)
    subdata = myPCA(subdata)
    myplot(subdata,label_states,kmeans.labels_,str(year_dict[key_period]))







print('aaa')
















