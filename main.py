
# Incluimos primero los paquetes
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product, permutations, chain
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from data import *
from entorno import *
import copy
import estimacion_M as eM
from auxiliar_functions import *
import networkx as nx
from MultiTargetPolygonal import MultiTargetPolygonal

np.random.seed(25)
# #
# # m = 10
#
# datos = Data([], m = 9,
#                  r = 4,
#                  capacity = 50,
#                  modo = 3,
#                  tmax = 900,
#                  init = 1,
#                  show = True,
#                  seed = 2)
# datos.generar_muestra()
#
# data = datos.mostrar_datos()

# ent1 = Poligonal(V=np.array([[3, 8], [6, 11], [11, 11]]), alpha= 1)
# ent2 = Poligonal(V=np.array([[3, 5], [1, 2], [1, 5]]), alpha= 1)
# ent3 = Poligonal(V=np.array([[5, 2], [7, 5], [12, 3]]), alpha = 1)
# ent4 = Punto(V = np.array([5, 7]))
# ent5 = Punto(V = np.array([9, 9]))
# ent6 = Punto(V = np.array([10, 7]))
# ent7 = Punto(V = np.array([19, 11]))
# ent8 = Poligonal(V=np.array([[14, 10], [16, 6], [19, 8], [17, 4], [12, 6]]), alpha = 0.5)


datos = Data([], m = 13,
                 r = 4,
                 capacity = 50,
                 vD = 2,
                 modo = 2,
                 tmax = 1800,
                 init = 3,
                 show = True,
                 seed = 2)

# 1047.45, 59.2%
# 989.524, 63.4%

# datos.data = [ent1, ent2, ent3], #ent4]

datos.generar_muestra()
#, ent5, ent6, ent7] #, ent8]

MultiTargetPolygonal(datos)
