import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

def myPCA(data):
    pca = PCA(n_components=20)
    data = pca.fit_transform(data)
    return data

def myStandard(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data[np.isnan(data)] = 0
    return data

def split_labels(labels, max_per_line=15):
    """将标签列表分成每行最多max_per_line个的多行文本"""
    lines = []
    for i in range(0, len(labels), max_per_line):
        line = ', '.join(labels[i:i+max_per_line])
        lines.append(line)
    return '<br>'.join(lines)  # 使用HTML的换行符来分行


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

    # 使用Plotly Express来生成散点图
    fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
                     title=figure_name,
                     color_discrete_sequence=colors,
                     hover_name='Label',
                     hover_data={'Cluster': False, 'PCA1': False, 'PCA2': False})

    # 准备左上角的注释文本
    annotations = []
    y_pos = 1.0
    vertical_spacing = 0.04  # 根据换行数量调整行间距

    for cluster in unique_clusters:
        labels = df[df['Cluster'] == cluster]['Label'].unique()
        label_text = split_labels(labels)  # 分行处理
        annotation_text = f"Cluster {cluster}: {label_text}"
        annotations.append(dict(
            xref='paper', yref='paper',
            x=0, y=y_pos,
            xanchor='left', yanchor='top',
            text=annotation_text,
            showarrow=False,
            align='left',
            bgcolor='rgba(0, 0, 0, 0)',  # 设置注释背景为透明
            font=dict(family='Arial', size=12, color=colors[unique_clusters.tolist().index(cluster)])
        ))
        # 更新下一个注释的y位置，根据行数动态调整
        y_pos -= vertical_spacing * (label_text.count('<br>') + 1)

        points = df[df['Cluster'] == cluster][['PCA1', 'PCA2']].values
        if points.shape[0] > 2:  # 凸包至少需要三点
            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
            y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])
            fig.add_trace(go.Scatter(x=x_hull, y=y_hull, mode='lines',
                                     line=dict(color=colors[unique_clusters.tolist().index(cluster)], width=2),
                                     showlegend=False))

    # 更新布局以添加注释和隐藏坐标轴标签
    fig.update_layout(
        annotations=annotations,
        xaxis={'visible': False, 'showticklabels': False},
        yaxis={'visible': False, 'showticklabels': False},
        legend_title_text='Cluster',
        coloraxis_showscale=False,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell")
    )

    # 显示图表
    fig.show()




year_dict = {
    'period1': list(range(1963, 1971)),  # 从1963到1970
    'period2': list(range(1971, 1981)),  # 从1971到1980
    'period3': list(range(1981, 1991)),
    'period4': list(range(1991, 2001)),
    'period5': list(range(2001, 2011)),
    'period6': list(range(2011, 2016)),
}


for key_year in year_dict.keys():
    ml_data = pd.read_csv(str('data/mldata_'+ key_year + '.csv'))
    ml_data.set_index(ml_data.columns[0], inplace=True, drop=True)
    ml_data.fillna(0, inplace=True)

    label_states = ml_data.index
    ml_data = myStandard(ml_data)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(ml_data)
    myplot(ml_data, label_states, kmeans.labels_, str(year_dict[key_year]))








