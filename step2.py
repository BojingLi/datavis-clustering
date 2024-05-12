import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
year_dict = {
    'period1': list(range(1963, 1971)),
    'period2': list(range(1971, 1981)),
    'period3': list(range(1981, 1991)),
    'period4': list(range(1991, 2001)),
    'period5': list(range(2001, 2011)),
    'period6': list(range(2011, 2016)),
}

stateList = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',
             'Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
             'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri',
             'Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina',
             'North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
             'South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia',
             'Wisconsin','Wyoming','District of Columbia']

for key_year in year_dict.keys():
    ml_data = pd.DataFrame
    for state in stateList:
        temp = pd.read_csv(str('data/'+ state + '_' + key_year + '.csv'))
        if ml_data.empty:
            ml_data = temp.copy()
        else:
            ml_data = pd.concat([ml_data, temp],ignore_index=True)

    saveFile = str('mldata_'+key_year +'.csv')
    path = os.path.join('data', saveFile)
    ml_data.set_index(ml_data.columns[0], inplace=True)
    ml_data.to_csv(path, index=True)

