import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def readDatabyState(stateName):
    df = pd.read_excel('Patent by state and category.xlsx', sheet_name=stateName)
    # df = pd.read_excel('testdata.xlsx', sheet_name=stateName)
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
            trans = period_data[statename].transpose()
            trans.columns = trans.iloc[0]
            trans = trans.drop("Class Title")
            trans.rename(index={'period_total': statename}, inplace=True)
            # 检查是否是第一次迭代
            if period_cluster.empty:
                period_cluster = trans  # 如果period_cluster是空的，直接赋值
            else:
                print("period_cluster="+period_cluster)
                print("trans=" + trans)

                # print('debug')
                period_cluster = pd.concat([period_cluster, trans],ignore_index=True)  # 否则，连接现有的数据
            period_cluster.fillna(0, inplace=True)
        mldata[key] = period_cluster
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


