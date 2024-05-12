import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import SpectralClustering
import plotly.express as px
import plotly.express as px
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np
import plotly.graph_objects as go



def myplot(data, plot_label, cluster_label, figure_name):
    # 创建DataFrame来用于Plotly
    df = pd.DataFrame({
        'PCA1': data[:, 0],
        'PCA2': data[:, 1],
        'Label': plot_label,
        'Cluster': cluster_label
    })

    df = df.sort_values(by='Cluster', ascending=True)
    df['Cluster'] = df['Cluster'].replace({0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})

    # 定义离散颜色列表
    unique_clusters = df['Cluster'].unique()
    colors = px.colors.qualitative.Plotly[:len(unique_clusters)]

    # 从fig.data中提取每个簇的颜色
    cluster_colors = {0:colors[0],1:colors[1],2:colors[2],3:colors[3],4:colors[4]}

    testcolor = px.colors.qualitative.Plotly[:len(unique_clusters)]


    # 使用Plotly Express来生成散点图
    fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
                     title=figure_name,
                     color_discrete_sequence=testcolor,
                     hover_name='Label',
                     hover_data={'Cluster': False})

    # 计算并绘制每个聚类的凸包
    for cluster in unique_clusters:
        points = df[df['Cluster'] == cluster][['PCA1', 'PCA2']].values
        if points.shape[0] > 2:  # 凸包至少需要三点
            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
            y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])
            fig.add_trace(go.Scatter(x=x_hull, y=y_hull, mode='lines',
                                     line=dict(color=colors[unique_clusters.tolist().index(cluster)], width=2),
                                     showlegend=False))

    # 更新布局以隐藏坐标轴标签和调整颜色条
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(xaxis={'visible': False, 'showticklabels': False},
                      yaxis={'visible': False, 'showticklabels': False},
                      legend_title_text='Cluster',
                      coloraxis_showscale=False,
                      hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell"))

    # 显示图表
    fig.show()



def myPCA(data):
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    return data

def myStandard(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data[np.isnan(data)] = 0
    return data






year_dict = {
    'period1': list(range(1963, 1971)),  # 从1963到1970
    'period2': list(range(1971, 1981)),  # 从1971到1980
    'period3': list(range(1981, 1991)),
    'period4': list(range(1991, 2001)),
    'period5': list(range(2001, 2011)),
    'period6': list(range(2011, 2016)),
}



ml_data = pd.read_csv(str('data/mldata_period1.csv'))
ml_data.set_index(ml_data.columns[0], inplace=True, drop=True)
ml_data.fillna(0, inplace=True)

label_states = ml_data.index
ml_data = myStandard(ml_data)
ml_data = myPCA(ml_data)


#kmeans聚类
# kmeans = KMeans(n_clusters=5, random_state=0).fit(ml_data)
# labels = kmeans.labels_

# 光谱聚类
spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=0)
spectral.fit(ml_data)
labels = spectral.labels_



key_year = 'test'
myplot(ml_data, label_states, labels, key_year)









