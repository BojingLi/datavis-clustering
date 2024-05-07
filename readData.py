import pandas as pd

def readDatabyState(stateName):
    # df = pd.read_excel('Patent by state and category.xlsx', sheet_name=stateName)
    df = pd.read_excel('testdata.xlsx', sheet_name=stateName)
    return df


# stateList = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',
#              'Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
#              'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri',
#              'Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina',
#              'North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
#              'South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia',
#              'Wisconsin','Wyoming','District of Columbia']



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
            period = [1963, 1964, 1965]
        elif key.endswith('period2'):
            period = [1966, 1967, 1968]
        else:
            period = [1969, 1970, 1971, 1972]

        for statename, statedata in value.items():
            statedata['period_total'] = statedata[period].sum(axis=1)
            statedata.drop(columns=period, inplace=True)
    return dict_data