import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import matplotlib
import tikzplotlib


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
matplotlib.rcParams['axes.unicode_minus'] = False

string = 'Point'

def create_picture(string):
    data = pd.read_excel('output_15_'+string+'.xlsx')
    
    # Selecting the elapsed lines
    rows_elapsed = data[(data['Elapsed'] == 'Elapsed')].index.tolist()
                
    
    # Selecting the times
    times = data.loc[rows_elapsed]['Objective']
    
    # Selecting the previous values
    rows_values = list(np.array(rows_elapsed)-1)
    
    # print(rows_elapsed)
    # print(rows_values)
    
    
    best_solution = data.loc[rows_values]['Solution']
    
    bound = data.loc[rows_values]['Bound']
    
    results = pd.DataFrame()
    
    results['Time'] = times
    results['Solution'] = list(best_solution)
    # print(results['Solution'][0:398])
    results['Bound'] = list(bound)
    results['Gap'] = (results['Solution'] - results['Bound'])/results['Solution']
    results['Type'] = string
    
    results.plot(x = 'Time', y = ['Solution', 'Bound'])
    # sns.lineplot(data = results, x = 'Time', y = 'Bound')
    
    plt.title(string)
    # plt.axis([0, 86400, 0, 1])
    plt.savefig('1day_'+string+'.png')
    tikzplotlib.save('1day_'+string+'.tex', encoding = 'utf-8')

    
    plt.clf()
    return results

point = create_picture('Point')
polygonal = create_picture('Polygonal')

gaps = pd.concat([point, polygonal])

# print(gaps['Gap2'])
g = sns.lineplot(data = gaps, x='Time', y = 'Gap', hue = 'Type')

# plt.axis([0, 86400, 0, 1])
tikzplotlib.save('gap_comparison.tex', encoding = 'utf-8')
# plt.show()


