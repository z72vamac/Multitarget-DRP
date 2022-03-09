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
from grafo_problem import grafo_problem
from clustering_easy import clustering_easy
from tsp import tsp
from aglomerativo import aglomerativo
from clusters import *

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

def heuristic(datos):
    # Segundo paso: Resolvemos utilizando grafo problem 
    
    first_time = time.time()
    
    nG = datos.m
    
    # paths = []
    #
    # for g in range(1, nG+1):
    #     path = grafo_problem(datos.data[g-1], datos.alpha, g)
    #     paths.append(path)
    #
    # print()
    # print(paths)
    # print()

    seeds = 10
    
    best_clusters = {}
    best_Ck = {}
    best_objval = 100000
    best_path = []
    
    for i in range(seeds):
        np.random.seed(i)
        
        # Tercer paso: Algoritmo aglomerativo para clustering_hard
        clusters = aglomerativo(datos)
        
        for i in clusters.keys():
            clusters[i].imprime()
        # Cuarto paso: Resolvemos la localizacion de los puntos una vez formado el clustering_hard
        # Ck = clustering_easy(datos, paths, clusters)
        
        # if Ck == None:
        #     continue
        Ck = []
        
        for cluster in clusters.values():
            if len(cluster.points) > 0:
                Ck.append(cluster.points[0])
                Ck.append(cluster.points[1])
        
        Ck_dict = {}
        
        for k in range(len(Ck)):
            Ck_dict[k] = Ck[k]
        #
        # Ck_dict = dict(Ck)
        
        print(Ck_dict)
            
        path, objVal = tsp(Ck_dict)
    
        if objVal <= best_objval:
            best_clusters = clusters
            best_objval = objVal
            best_Ck = Ck
            best_path = path
    
    
    mugt_sol = []
    
    
    # for i, t in zip(best_path, range(1, len(best_path)+1)):
    for t in best_clusters.keys():
        for g in best_clusters[t].indices:
            # e1 = paths[g-1][0][0]
            # e2 = paths[g-1][0][-1]
            mugt_sol.append((g, t))
            
    print(mugt_sol)
    # print(vigtd_sol)
    
    with open('solution_heuristic.txt', 'a') as f:
    
        f.write('Coordinates of the graphs squares. V[')
        
        for g, it in zip(datos.data, range(len(datos.data))):
            for i, it2 in zip(g.V, range(len(g.V))):
                f.write('V[{0}, {1}] = {2}\n'.format(it2, it, i))
                
        f.write('Centroids: ' + str(best_Ck) + '\n')
        f.write('Clusters labels: ' + str(best_clusters) + '\n')
        f.write('Path labels: ' + str(best_path) + '\n')
    
    # z_sol = []
    #
    # for lista, it in zip(paths, range(1, len(paths)+1)):
    #     for i in range(len(lista[0])-1):
    #         z_sol.append((lista[0][i], lista[0][i+1], it))
    
    # with open('./case_study/z.')
        # f.close('solution_heuristic.txt', 'a')
    
    # f.write(best_Ck)
    
    print(best_Ck)
    
    for i in best_clusters.keys():
        best_clusters[i].imprime()
        
    # print(best_clusters)
    # print(best_objval)
    # print(paths)
    # print(z_sol)
    print(best_path)
    
    # fig, ax = plt.subplots()
    #
    # for g in range(1, datos.m+1):
        # grafo = datos.data[g-1]
        # centroide = np.mean(grafo.V, axis = 0)
        # nx.draw(grafo.G, grafo.pos, node_size=100, node_color='black', alpha=1, width = 1, edge_color='black')
        # ax.annotate(g, xy = (centroide[0], centroide[1]))
        # nx.draw_networkx_labels(grafo.G, grafo.pos, font_color = 'red', font_size=9)
        #
    # colores = ['blue', 'green', 'red', 'purple', 'black', 'yellow', 'brown']
    #
    # for (k, i) in zip(best_clusters.keys(), range(len(best_clusters.keys()))):
        # plt.plot(best_Ck[(k, 0)], best_Ck[(k, 1)], 'ko', color = colores[i])
        # ax.annotate(k, xy = (best_Ck[(k, 0)], best_Ck[(k, 1)]))
        #
    # plt.show()
    
    second_time = time.time()
    
    heuristic_time = second_time - first_time
    

    return mugt_sol, heuristic_time

# heuristic(datos)

