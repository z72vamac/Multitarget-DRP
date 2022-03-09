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
import time
from MTZ import MTZ

# np.random.seed(3)
# datos = Data([], m = 6,
#                  r = 2,
#                  capacity = 50,
#                  modo = 3,
#                  tmax = 900,
#                  init = 0,
#                  show = True,
#                  seed = 2)
# datos.generar_muestra()
#
# data = datos.mostrar_datos()

def heuristic2(datos):
    # Segundo paso: Resolvemos utilizando grafo problem 
    
    first_time = time.time()
    
    nG = datos.m
    
    
    
    
    second_time = time.time()
    
    heuristic_time = second_time - first_time
    

    return mugt_sol, heuristic_time

# heuristic(datos)

