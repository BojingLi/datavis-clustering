import pandas as pd
import plotly.express as px

# 创建一个示例 DataFrame
data = {
    'PCA1': [1, 2, 3, 4, 5],
    'PCA2': [2, 3, 4, 5, 6],
    'Cluster': ['0', '1', '0', '0', '1'],
    'Label': ['Point 1', 'Point 2', 'Point 3', 'Point 4', 'Point 5']
}
df = pd.DataFrame(data)

# 定义颜色序列
testcolor = ['red', 'black']

# 创建散点图
fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
                 title='Scatter Plot with Color by Cluster',
                 color_discrete_sequence=testcolor,
                 hover_name='Label',
                 hover_data={'Cluster': False})

# 显示图形
fig.show()
