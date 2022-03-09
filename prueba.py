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
from auxiliar_functions import path2matrix, min_dist
from preprocessing3 import preprocessing3
import auxiliar_functions as af
from heuristic import heuristic
from MTZ import MTZ


edges = gp.tuplelist(((3, 5), (5, 4), (4, 3)))

def subtour(edges):
    "Genera un subtour de una lista de aristas"
    
    # edges = [(1, 2), (2, 4), (4, 1)]
    unvisited = []
    for i, j in edges:
        unvisited.append(i)
        unvisited.append(j)
    
    unvisited = list(dict.fromkeys(unvisited))
    
    # unvisited = [1, 2, 4]
    
    #cycle = range(m)  # initial length has 1 more city
    if len(unvisited) == len(edges):
        cycle = range(len(unvisited)+1)
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            # neighbors = [1, 2, 4]
            while neighbors:
                current = neighbors[0]
                # current = 1
                thiscycle.append(current)
                # thiscycle = [1]
                unvisited.remove(current)
                # unvisted = [2, 4]
                neighbors = [j for i, j in edges.select(current, '*')
                             if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle
    else:
        return []      

cycle = subtour(edges)
print(cycle)