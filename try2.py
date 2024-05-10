import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
import numpy as np
from functools import reduce
# warnings.filterwarnings('ignore')



def readDatabyState(stateName):
    df = pd.read_excel('Patent by state and category.xlsx', sheet_name=stateName)
    # df = pd.read_excel('testdata.xlsx', sheet_name=stateName)
    df.drop(df.index[-1], inplace=True)
    return df

def splitDataByYear(data_dict, year_dict):
    # 按照年份截取
    split_state_all_period = {}
    for period_name, period_data in year_dict.items():
        split_state_period = {}  # 创建一个新的字典来存储当前周期的数据
        selected_columns = ['Class', 'Class Title'] + period_data
        for state_name, state_data in data_dict.items():
            # 从每个州的数据中选取特定的列
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

def makeClusterData(datadict,yeardict):
#     把每一个时期数据转化为机器学习标准数据，每一行一个州，每一列一个特征（专利类型）
    mldata = {}
    for key in yeardict:
        # 每一个period中，州的数据聚合成可供聚类的数据
        period_data = datadict[key]
        period_cluster = pd.DataFrame()
        for statename, statedata in period_data.items():
            period_data[statename].rename(columns={'period_total': statename}, inplace=True)
            if period_cluster.empty:
                period_cluster = period_data[statename].copy()  # 如果period_cluster是空的，直接赋值
            else:
                period_cluster = (pd.merge(period_cluster, period_data[statename], on="Class Title", how="outer")).fillna(0)
        mldata[key] = period_cluster
    return mldata


def makeClusterData2(datadict, yeardict):
    mldata = {}
    for key in yeardict:
        frames = []
        for statename, statedata in datadict[key].items():
            statedata.rename(columns={'period_total': statename}, inplace=True)
            frames.append(statedata.set_index('Class Title'))

        # 使用 concat 合并所有的 DataFrame，这里假设所有的 DataFrame 都已经按 'Class Title' 设置了索引
        if frames:
            period_cluster = pd.merge(frames, axis=1, sort=False).fillna(0)
            mldata[key] = period_cluster.reset_index()
    return mldata


def makeClusterData3(datadict, yeardict):
    mldata = {}
    for key in yeardict:
        # 每一个period中，州的数据聚合成可供聚类的数据
        period_data = datadict[key]
        frames = []  # 用于存放所有需要合并的 DataFrame
        for statename, statedata in period_data.items():
            # 重命名 'period_total' 列为州名
            statedata.rename(columns={'period_total': statename}, inplace=True)
            # 设置 'Class Title' 为索引，便于合并
            statedata.set_index('Class Title', inplace=True)
            frames.append(statedata)

        # 使用 reduce 和 pd.merge 一次性合并所有 DataFrame
        if frames:
            period_cluster = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'),
                                    frames).fillna(0)
            # 重设索引，如果需要的话
            period_cluster.reset_index(inplace=True)
            mldata[key] = period_cluster
        else:
            mldata[key] = pd.DataFrame()
    return mldata


def myPCA(data):
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    return data

def myStandard(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

def myplot(data,plot_label,color_label,figure_name):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=color_label, cmap='viridis', alpha=0.5)
    # 在散点图中为每个点添加名字标签
    for i, txt in enumerate(plot_label):
        plt.annotate(txt, (data[i, 0], data[i, 1]))
    # 移除坐标轴刻度
    plt.xticks([])
    plt.yticks([])

    plt.title(figure_name)
    colorbar = plt.colorbar(scatter)
    colorbar.set_ticks([])

    # plt.show()
    # 保存图像到本地文件系统
    plt.savefig(f"{figure_name}.png")
    # 关闭图形以避免其显示
    plt.close()


#读取全部州的数据
raw_state_data = {}
# stateList = ['Alabama','Alaska','Arizona','Arkansas']
stateList = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',
             'Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
             'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri',
             'Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina',
             'North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
             'South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia',
             'Wisconsin','Wyoming','District of Columbia']


for state in stateList:
    raw_state_data[state]  = readDatabyState(state)


#把数据按照时期聚合
year_dict = {
    'period1': list(range(1963, 1971)),  # 从1963到1970
    'period2': list(range(1971, 1981)),  # 从1971到1980
    'period3': list(range(1981, 1991)),
    'period4': list(range(1991, 2001)),
    'period5': list(range(2001, 2011)),
    'period6': list(range(2011, 2016)),
}
split_state_all_period = splitDataByYear(raw_state_data,year_dict)
aggr_state_all_period = aggrDataByYear(split_state_all_period)

# 进行机器学习聚类操作
ml_data = makeClusterData3(aggr_state_all_period,year_dict)
# for key_period,value_data in ml_data.items():
#     subdata = ml_data[key_period]
#
#     # 保存州的索引,最后添加到每个聚类的圆点旁边
#     label_states = subdata.index
#
#     subdata = myStandard(subdata)
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(subdata)
#     subdata = myPCA(subdata)
#     myplot(subdata,label_states,kmeans.labels_,str(year_dict[key_period]))
#
#
#
#
#
#
#
print('aaa')



