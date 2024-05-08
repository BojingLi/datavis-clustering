

import pandas as pd
from readData import readDatabyState
from readData import splitDataByYear
from readData import aggrDataByYear



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

# 每一个period中，州的数据聚合成可供聚类的数据
period_data = aggr_state_all_period['period1']
period_cluster = pd.DataFrame()
for statename, statedata in period_data.items():
    trans = period_data[statename].transpose()
    trans.columns = trans.iloc[0]
    trans = trans.drop("Class Title")
    trans.rename(index={'period_total': statename}, inplace=True)

    # 检查是否是第一次迭代
    if period_cluster.empty:
        period_cluster = trans  # 如果period_cluster是空的，直接赋值
    else:
        period_cluster = pd.concat([period_cluster, trans])  # 否则，连接现有的数据



print('aaa')
















