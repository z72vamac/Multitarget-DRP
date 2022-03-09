import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

pd.set_option('display.max_rows', 500)

# point_gurobi = pd.read_csv('results_point_gurobi_with_initialization.csv')
# point_gurobi = point_gurobi[['Size', 'Instance', 'Capacity', 'GAP', 'NodeCount', 'Runtime', 'ObjVal', 'HeurTime', 'HeurVal']]
#
# point_gurobi_gettingtime = pd.read_csv('results_point_gurobi_with_initialization_gettingtime.csv')
# point_gurobi['HeurTime'] = point_gurobi_gettingtime['HeurVal']
# # point_gurobi['Runtime'] = np.log(point_gurobi['Runtime'])
# # point_gurobi['Type'] = 'Point'
# point_gurobi.to_csv('results_point_gurobi_with_initialization_corrected.csv')
#
# polygonal_gurobi = pd.read_csv('results_polygonal_gurobi_with_initialization.csv')
# polygonal_gurobi = polygonal_gurobi[['Size', 'Instance', 'Capacity', 'GAP', 'NodeCount', 'Runtime', 'ObjVal', 'HeurTime', 'HeurVal']]
#
# polygonal_gurobi_gettingtime = pd.read_csv('results_polygonal_gurobi_with_initialization_gettingtime.csv')
# polygonal_gurobi['HeurTime'] = polygonal_gurobi_gettingtime['HeurVal']
# # polygonal_gurobi['Runtime'] = np.log(polygonal_gurobi['Runtime'])
# # polygonal_gurobi['Type'] = 'Polygonal'
# polygonal_gurobi.to_csv('results_polygonal_gurobi_with_initialization_corrected.csv')

point_cplex = pd.read_csv('results_point_cplex_with_initialization.csv')
point_cplex = point_cplex[['Size', 'Instance', 'Capacity', 'GAP', 'NodeCount', 'Runtime', 'ObjVal', 'HeurTime', 'HeurVal']]

point_cplex_gettingtime = pd.read_csv('results_point_cplex_with_initialization_gettingtime.csv')
point_cplex['HeurTime'] = point_cplex_gettingtime['Time_h']
point_cplex['HeurVal'] = point_cplex_gettingtime['ObjVal_h']
# point_cplex['Runtime'] = np.log(point_cplex['Runtime'])
# point_cplex['Type'] = 'Point'
point_cplex.to_csv('results_point_cplex_with_initialization_corrected.csv')

# polygonal_cplex = pd.read_csv('results_polygonal_cplex_with_initialization.csv')
# polygonal_cplex = polygonal_cplex[['Size', 'Instance', 'Capacity', 'GAP', 'NodeCount', 'Runtime', 'ObjVal', 'HeurTime', 'HeurVal']]
#
# polygonal_cplex_gettingtime = pd.read_csv('results_polygonal_cplex_with_initialization_gettingtime.csv')
# polygonal_cplex['HeurTime'] = polygonal_cplex_gettingtime['Time_h']
# polygonal_cplex['HeurVal'] = polygonal_cplex_gettingtime['ObjVal_h']
# polygonal_cplex.to_csv('results_polygonal_cplex_with_initialization_corrected.csv')
#
# tabla_comparadora.to_excel('summary_gurobi_without_initialization.xlsx')
# print(tabla_comparadora)
#
# pd.set_option('display.max_rows', 500)
#
# point_cplex = pd.read_csv('results_point_cplex_without_initialization.csv')
# point_cplex = point_cplex[['Size', 'Instance', 'Capacity', 'GAP', 'NodeCount', 'Runtime', 'ObjVal']]
# # point_cplex['Runtime'] = np.log(point_cplex['Runtime'])
# point_cplex['Type'] = 'Point'
#
# polygonal_cplex = pd.read_csv('results_polygonal_cplex_without_initialization.csv')
# polygonal_cplex = polygonal_cplex[['Size', 'Instance', 'Capacity', 'GAP', 'NodeCount', 'Runtime', 'ObjVal']]
# # polygonal_cplex['Runtime'] = np.log(polygonal_cplex['Runtime'])
# polygonal_cplex['Type'] = 'Polygonal'
#
# comparador = pd.concat([point_cplex, polygonal_cplex])
# comparador[['Size', 'Instance', 'Capacity']] = comparador[['Size', 'Instance', 'Capacity']].apply(np.int64)
# tabla_comparadora = comparador.groupby(['Type', 'Size', 'Capacity']).describe()[['GAP', 'Runtime']].round(2).reset_index()
# number_nan = comparador.groupby(['Size', 'Capacity']).isna().sum()[['GAP', 'Runtime']].round(2).reset_index()

# tabla_comparadora.to_excel('summary_cplex_without_initialization.xlsx')
# print(tabla_comparadora)
