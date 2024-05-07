

import pandas as pd
from readData import readDatabyState
from readData import splitDataByYear
from readData import aggrDataByYear



#读取全部州的数据
raw_state_data = {}
stateList = ['Alabama','Alaska','Arizona','Arkansas']
for state in stateList:
    raw_state_data[state]  = readDatabyState(state)


#把数据按照年份聚合
year_dict = {
    'period1': [1963, 1964, 1965],
    'period2': [1966, 1967, 1968],
    'period3': [1969, 1970, 1971, 1972]
}
split_state_all_period = splitDataByYear(raw_state_data,year_dict)
aggr_state_all_period = aggrDataByYear(split_state_all_period)





print('aaa')
















