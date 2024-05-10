import pandas as pd


aaa = pd.read_csv('aaa.csv')
bbb = pd.read_csv('bbb.csv')

result = pd.concat([aaa, bbb],ignore_index=True)




print('end')

