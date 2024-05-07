

import pandas as pd


    df = pd.read_excel('Patent by state and category.xlsx', sheet_name=stateName)



stateList = ['Alabama','Alaska']


# stateList = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',
#              'Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
#              'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri',
#              'Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina',
#              'North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
#              'South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia',
#              'Wisconsin','Wyoming','District of Columbia']

for state in stateList:
    # locals()[state] = readDatabyState(state)
    df = readDatabyState(state)






print('end')















