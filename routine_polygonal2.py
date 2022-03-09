import gurobipy as gp
import pdb
from gurobipy import GRB
import numpy as np
from itertools import product
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
from data import *
from entorno import *
import copy
import estimacion_M as eM
import networkx as nx
import auxiliar_functions as af
from MultiTargetPolygonal import MultiTargetPolygonal
import csv
import pandas as pd
import pickle as pickle
from avisador import avisador

instancias = pickle.load(open("instancias_poligonales.pickle", "rb"))
# instancias_deulonay = pickle.load(open("instancias_deulonay.pickle", "rb"))

dataframe = pd.DataFrame(columns = ['Instance', 'Size', 'Capacity', 'GAP', 'Runtime', 'NodeCount', 'ObjVal', 'HeurTime', 'HeurVal'])
iter = 0

for key, it in zip(instancias.keys(), range(len(instancias.keys()))):
    
    instance, size, capacity = key
    datos = instancias[key]
    if size >= 5:# and size <= 15 and iter >= 96 + 130: #and iter >= 26 + 9: #and iter >= 188 + 13:

        datos.init = True
    
        print()
        print('--------------------------------------------')
        print('MultiTargetPolygonal: Instance: {a} - Size: {b} - Capacity: {c}'.format(a = instance, b = size, c = capacity))
        print('--------------------------------------------')
        print()
            
        solution = MultiTargetPolygonal(datos, 10)
        
        dataframe = dataframe.append(pd.Series([instance, size, capacity, solution[0], solution[1], solution[2], solution[3], solution[4], solution[5]], index=['Instance', 'Size', 'Capacity', 'GAP', 'Runtime', 'NodeCount', 'ObjVal', 'HeurTime', 'HeurVal']), ignore_index=True)
    
        dataframe.to_csv('results_polygonal_gurobi_with_initialization_Test.csv', header = True, mode = 'w')
        
        # avisador(iter, 'polygonal with initialization')
    iter += 1