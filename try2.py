import pandas as pd

# 假设df1, df2, ... 是你的DataFrame，例如：
data1 = {
    'Drug, Bio-Affecting and Bod...': [2],
    'Measuring and Testing': [22],
    'Organic Compounds (includ...': [12]
}
df1 = pd.DataFrame(data1, index=["Alabama"])

data2 = {
    'Drug, Bio-Affecting and Bod...': [5],
    'Measuring and Testing': [25],
    'Organic Compounds (includ...': [15]
}
df2 = pd.DataFrame(data2, index=["Alaska"])

# 增加更多DataFrame，只需确保所有DataFrame的列名相同

# 使用concat合并DataFrame
combined_df = pd.concat([df1, df2])  # 可以在列表中加入更多的df

# 查看合并后的结果
print(combined_df)
