import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
import numpy as np
from functools import reduce
import os
import warnings
warnings.filterwarnings('ignore')

def readDatabyState(stateName):
    df = pd.read_excel('Patent by state and category.xlsx', sheet_name=stateName)
    # df = pd.read_excel('testdata.xlsx', sheet_name=stateName)
    df.drop(df.index[-1], inplace=True)
    return df

def splitDataByYear(data_dict, year_dict):
    # 按照年份截取
    split_state_all_period = {}
    for period_name, period_data in year_dict.items():
        split_state_period = {}  # Create a new dictionary to store data for the current period
        selected_columns = ['Class', 'Class Title'] + period_data
        for state_name, state_data in data_dict.items():
            # Select specific columns from each state's data
            split_state_period[state_name] = state_data[selected_columns]
        split_state_all_period[period_name] = split_state_period

    return split_state_all_period

def aggrDataByYear(dict_data):
    for key, value in dict_data.items():
        if key.endswith('period1'):
            period = list(range(1963, 1971))
        elif key.endswith('period2'):
            period = list(range(1971, 1981))
        elif key.endswith('period3'):
            period = list(range(1981, 1991))
        elif key.endswith('period4'):
            period = list(range(1991, 2001))
        elif key.endswith('period5'):
            period = list(range(2001, 2011))
        else:
            period = list(range(2011, 2016))

        for statename, statedata in value.items():
            statedata['period_total'] = statedata[period].sum(axis=1)
            statedata.drop(columns=period, inplace=True)
            statedata.drop(columns='Class', inplace=True)
    return dict_data



def saveDatatoCluster(datadict,yeardict):
    if not os.path.exists('data'):
        os.makedirs('data')
#     Convert each period of data into machine learning standard data,
    #     one state per row, one feature per column (patent class)
    mldata = {}
    for key in yeardict:
        # In each period, state data is aggregated into data available for clustering
        period_data = datadict[key]
        period_cluster = pd.DataFrame()
        for statename, statedata in period_data.items():
            trans = period_data[statename].transpose()
            trans.columns = trans.iloc[0]
            trans = trans.drop("Class Title")
            trans.rename(index={'period_total': statename}, inplace=True)

            # 保存文件
            filename = str(trans.index[0]) + '_'+ key+ '.csv'
            path = os.path.join('data', filename)
            trans.to_csv(path, index=True)


#Read all states
raw_state_data = {}
stateList = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',
             'Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
             'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri',
             'Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina',
             'North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
             'South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia',
             'Wisconsin','Wyoming','District of Columbia']


for state in stateList:
    raw_state_data[state]  = readDatabyState(state)


#Aggregate data by period
year_dict = {
    'period1': list(range(1963, 1971)),
    'period2': list(range(1971, 1981)),
    'period3': list(range(1981, 1991)),
    'period4': list(range(1991, 2001)),
    'period5': list(range(2001, 2011)),
    'period6': list(range(2011, 2016)),
}

split_state_all_period = splitDataByYear(raw_state_data,year_dict)
aggr_state_all_period = aggrDataByYear(split_state_all_period)
saveDatatoCluster(aggr_state_all_period,year_dict)