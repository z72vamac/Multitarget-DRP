#!/usr/bin/python

# Copyright 2019, Gurobi Optimization, LLC

# Solve the classic diet model, showing how to add constraints
# to an existing model.

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
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from entorno import Poligono, Elipse, Poligonal, Punto
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import estimacion_M as eM
from data import Data
from itertools import combinations
import auxiliar_functions as af
import time
from heuristic import heuristic

# np.random.seed(2)
#
# datos = Data([], m = 10,
#                  r = 4,
#                  capacity = 40,
#                  modo = 1,
#                  tmax = 100,
#                  init = 2,
#                  show = True,
#                  seed = 2)
# datos.generar_muestra()
#
# data = datos.mostrar_datos()

# #
# prueba = 1
def MTZ2(datos):

    data = datos.mostrar_datos()

    # if datos.init:
    #     ps, ds, zs, dcs, scs, ss, us, obj_heur, res_time_heur, path_P = approach_MTZ(
    #         datos)
    start_time = time.time()

    """
    Input:
        data = lista de componentes
        init = si se proporciona valores iniciales
        datos.tmax = tiempo maximo que se le proporciona al modelo
        datos.show = si queremos que se imprima los caminos
    """

    m = len(data)

    modo = datos.modo

    # Model
    M = gp.Model("MTZ_MODEL")

    # Generando variables continuas de entrada y salida
    x_index = []
    d_index = []

    for c in range(m):
        # comp = data[c]
        # if type(comp) is Poligonal:
        x_index.append((c, 0, 0))
        x_index.append((c, 0, 1))
        x_index.append((c, 1, 0))
        x_index.append((c, 1, 1))
        d_index.append(c)

    x = M.addVars(x_index, vtype=GRB.CONTINUOUS, name="x")

    z_index = []

    for c1 in d_index:
        for c2 in d_index:
            if c1 != c2:
                z_index.append((c1, c2))

    p = M.addVars(z_index, vtype=GRB.CONTINUOUS, lb=0, name="p")

    z = M.addVars(z_index, vtype=GRB.BINARY, name='z')
    d = M.addVars(z_index, vtype=GRB.CONTINUOUS, lb=0, name='dcc')

    # if datos.init:
    #     d = M.addVars(z_index, vtype = GRB.CONTINUOUS, lb = ds, name = 'dcc')

    dif = M.addVars(z_index, 2, vtype=GRB.CONTINUOUS, lb=0, name='dif')

    dc = M.addVars(d_index, vtype=GRB.CONTINUOUS, lb=0, name='dc')
    difc = M.addVars(d_index, 2, vtype=GRB.CONTINUOUS, lb=0, name='difc')

    # Generando los mus de la envolvente convexa, los landas de la poligonal y las
    # variables binarias que indican qué segmento se elige

    mu_index = []
    landa_index = []
    sublanda_index = []
    u_index = []
    s_index = []

    for c in d_index:
        comp = data[c]
        if type(comp) is Poligono:
            for mu in range(comp.num_puntos):
                mu_index.append((c, mu))
        if type(comp) is Poligonal:
            u_index.append(c)
            # landa de la variable de entrada en la poligonal c
            landa_index.append((c, 0))
            # landa de la variable de salida en la poligonal c
            landa_index.append((c, 1))
            for segm in range(comp.num_segmentos):
                s_index.append((c, 0, segm))
                s_index.append((c, 1, segm))
            for punto in range(comp.num_puntos):
                sublanda_index.append((c, 0, punto))
                sublanda_index.append((c, 1, punto))

    mu = M.addVars(mu_index, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='mu')
    landa = M.addVars(landa_index, vtype=GRB.CONTINUOUS, name='landa')
    sublanda = M.addVars(sublanda_index, vtype=GRB.CONTINUOUS,
                         lb=0.0, ub=1.0, name='sublanda')
    s = M.addVars(s_index, vtype=GRB.BINARY, name='s')
    u = M.addVars(u_index, vtype=GRB.BINARY, name='u')
    minc = M.addVars(u_index, vtype=GRB.CONTINUOUS, lb=0.0, name='min')
    maxc = M.addVars(u_index, vtype=GRB.CONTINUOUS, lb=0.0, name='max')

    sc = M.addVars(d_index, vtype=GRB.CONTINUOUS, lb=0, ub=m - 1, name='sc')

    #
    #     for c in sc.keys():
    #         sc[c].start = scs[c]
#            d[i, j].start = ds[i, j]

    # for c in s.keys():
    #     s[c].start = ss[c]
    #
    # for c in u.keys():
    #     u[c].start = us[c]
#        for c in dc.keys():
#            dc[c].start = dcs[c]

    # Constraints
    for c1, c2 in z.keys():
        comp1 = data[c1]
        comp2 = data[c2]

        BigM = eM.estima_BigM_local(comp1, comp2)
        SmallM = eM.estima_SmallM_local(comp1, comp2)
            # M.addConstr(d[c1, c2] <= BigM)
            # M.addConstr(d[c1, c2]>= SmallM)
            # SmallM, x_0, x_1 = af.min_dist(comp1, comp2)
        M.addConstr(p[c1, c2] >= SmallM * z[c1, c2], name='p2')
        M.addConstr(p[c1, c2] >= d[c1, c2] - BigM * (1 - z[c1, c2]), name='p3')
        M.addConstr(d[c1, c2] <= BigM)
            #
            # M.addConstr(p[c1, c2] >= SmallM*z[c1, c2])
            # M.addConstr(p[c1, c2] <= d[c1, c2] + z[c1, c2]*SmallM - SmallM)
            # M.addConstr(p[c1, c2] <= z[c1, c2]*BigM)
    
        M.addConstr(dif[c1, c2, 0] >= x[c2, 0, 0] - x[c1, 1, 0])
        M.addConstr(dif[c1, c2, 0] >= -x[c2, 0, 0] + x[c1, 1, 0])
        M.addConstr(dif[c1, c2, 1] >= x[c2, 0, 1] - x[c1, 1, 1])
        M.addConstr(dif[c1, c2, 1] >= -x[c2, 0, 1] + x[c1, 1, 1])

        M.addConstr(dif[c1, c2, 0] * dif[c1, c2, 0] +
                    dif[c1, c2, 1] * dif[c1, c2, 1] <= d[c1, c2] * d[c1, c2])



    # Restriccion 2
    M.addConstrs(z.sum(c, '*') == 1 for c in range(m))

    # Restriccion 3
    M.addConstrs(z.sum('*', c) == 1 for c in range(m))

    # Restriccion 7
    for c1, c2 in z.keys():
        M.addConstr(z[c1, c2] + z[c2, c1] <= 1, name = 'rest7')

    # Restriccion 8
    for c1 in range(1, m):
        for c2 in range(1, m):
            if c1 != c2:
                M.addConstr(m - 1 >= (sc[c1] - sc[c2]) + m * z[c1, c2])
                # M.addConstr(sc[c1] - sc[c2] + m*z[c1, c2] + (m - 2)*z[c2, c1] <= m - 1)

    # Restriccion 10
    M.addConstr(sc[0] == 0, name = 'rest10')

    for c in range(1, m):
        M.addConstr(sc[c] >= 1, name = 'rest11')

    prueba = 0
    if prueba:
        # Restriccion 12
        for j in range(1, m):
            M.addConstr(sc[j] - sc[0] + (m - 2)*z[0, j] <= m - 1, name = 'rest12')

        # Restriccion 13
        for i in range(1, m):
            M.addConstr(sc[0] - sc[i] + (m - 1)*z[i, 0] <= 0, name = 'rest13')

        # # Restriccion 14
        # for i in range(1, m):
        #     for j in range(1, m):
        #         if i != j:
        #             M.addConstr(1-m+m*z[i, j] <= sc[j] - sc[i])
        #
        # # Restriccion 14
        # for i in range(1, m):
        #     for j in range(1, m):
        #         if i != j:
        #             M.addConstr(sc[j] - sc[i] <= m - 1 - (m-2)*z[i, j])

        # Restriccion 15
        for i in range(m):
            for j in range(1, m):
                if i != j:
                    M.addConstr(sc[i] - sc[j] + m*z[i, j] <= m - 1)

        # Restriccion 16
        for i in range(1, m):
            for j in range(m):
                if i != j:
                    M.addConstr(sc[i] - sc[j] + (m-2)*z[j, i] <= m - 1)

        for i in range(1, m):
            M.addConstr(-z[0,i] - sc[i] + (m-3)*z[i,0] <= -2, name="LiftedLB(%s)"%i)
            M.addConstr(-z[i,0] + sc[i] + (m-3)*z[0,i] <= m-2, name="LiftedUB(%s)"%i)

    # if datos.init:
    #     for c1, c2 in z.keys():
    #         M.addConstr(z[c1, c2] == zs[c1, c2])

    for c in range(m):
        comp = data[c]
        if type(comp) is not Poligonal:
            BigM_inside = eM.estima_max_inside(comp)
            M.addConstr(dc[c] <= BigM_inside)
        if type(comp) is Poligonal:
            M.addConstr(landa[c, 0] - landa[c, 1] == maxc[c] - minc[c], name='u0')
            # si u = 0, entonces landa0 >= landa1
            M.addConstr(maxc[c] + minc[c] >= comp.alpha * comp.num_segmentos, name='u1')
            M.addConstr(maxc[c] <= comp.num_segmentos * (1 - u[c]), name='u2')
            M.addConstr(minc[c] <= comp.num_segmentos * u[c], name='u3')
            M.addConstr(dc[c] == (maxc[c] + minc[c]) * comp.longitud / comp.num_segmentos)
            for i in range(2):
                for punto in range(1, comp.num_puntos):
                    M.addConstr(landa[c, i] - punto >= sublanda[c, i, punto] - comp.num_puntos * (1 - s[c, i, punto - 1]))
                    M.addConstr(landa[c, i] - punto <= sublanda[c, i, punto] + comp.num_puntos * (1 - s[c, i, punto - 1]))
                M.addConstr(sublanda[c, i, 0] <= s[c, i, 0])
                M.addConstr(sublanda[c, i, comp.num_puntos - 1]
                            <= s[c, i, comp.num_puntos - 2])
                for punto in range(1, comp.num_puntos - 1):
                    M.addConstr(sublanda[c, i, punto] <=
                                s[c, i, punto - 1] + s[c, i, punto])
                M.addConstr(s.sum(c, i, '*') == 1)
                M.addConstr(sublanda.sum(c, i, '*') == 1)
                for j in range(2):
                    M.addConstr(x[c, i, j] ==
                                gp.quicksum(sublanda[c, i, punto] * comp.V[punto][j] for punto in range(comp.num_puntos)), name='seg1')
                    
        if type(comp) is Punto:
                M.addConstrs(x[c, i, dim] == comp.V[dim] for i in range(2) for dim in range(2))
                
    M.update()

    fc = np.ones(m) * 1

    # for c in range(m):
    #     if fc[c] >= 1.0 and type(data[c]) is not Poligonal:
    #         for j in range(2):
    #             M.addConstr(x[c, 0, j] == x[c, 1, j])

    objective = gp.quicksum(p[c1, c2] for c1 in range(m) for c2 in range(m) if c1 != c2) + gp.quicksum(fc[c] * dc[c] for c in range(m))

    # objective = gp.quicksum(p[index] for index in p.keys())

    M.setObjective(objective, GRB.MINIMIZE)

    M.update()

    #M.addConstr(M.getObjective() <= obj_heur)
    #M.addConstr(M.getObjective() >= suma)


    # if datos.tmax is not None:
    #     M.Params.timeLimit = datos.tmax

    if not(datos.show):
        M.setParam('OutputFlag', 0)

    M.Params.timeLimit = 30
    M.Params.Threads = 8
    M.Params.tuneTimeLimit = 30
    # M.tune()
    M.Params.Method = 4
    # M.Params.Heuristic = 0
    # M.Params.IntFeasTol = 1e-1
    # M.Params.NonConvex = 2
    # M.Params.FeasibilityTol = 1e-6

    M.update()

    # M.write('./instances/MTZ-{i}-{j}-{k}.lp'.format(i = datos.m, j = datos.r, k = datos.modo))

    # Solve
    M.optimize()


    resultados = []

    if M.Status == 3:
        M.computeIIS()
        M.write('casa.ilp')
        datos.dibujar_muestra()
        gap = 100
        nodeCount = M.getAttr('NodeCount')
        print('Tiempo de resolución: ' + str(end_time - start_time))
        print('Gap restante: ' + str(gap))
        print('Número de nodos visitados: ' + str(nodeCount))

        plt.title(str(m) + "- XPPN - Mode " + str(modo))
        string = 'imagenes/' + \
            str(m) + "-XPPN - MTZ - Mode " + str(modo) + "infactible.png"
        plt.savefig(string)
        plt.close()

        resultados.append(end_time - start_time)
        resultados.append(gap)
        resultados.append(nodeCount)

        # if datos.init:
        #     resultados.append(np.nan)
        #     resultados.append(obj_heur)
        #     resultados.append(res_time_heur)

        return resultados

    if M.getAttr('SolCount') == 0:
        datos.dibujar_muestra()
        gap = 100
        nodeCount = M.getAttr('NodeCount')
        obj_model = np.nan
        print('No se ha hallado ninguna solución inicial')
        print('Tiempo de resolución: ' + str(end_time - start_time))
        print('Gap restante: ' + str(gap))
        print('Número de nodos visitados: ' + str(nodeCount))
        print('Número de restricciones de corte: ' + str(np.nan))

        resultados.append(end_time - start_time)
        resultados.append(gap)
        resultados.append(nodeCount)

        # if datos.init:
        #     resultados.append(np.nan)
        #     resultados.append(obj_heur)
        #     resultados.append(res_time_heur)

        return resultados

    vals = M.getAttr('x', z)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    tour = af.subtour(selected)

    if len(tour) < m:
        gap = M.getAttr('MIPGap')
        nodeCount = M.getAttr('NodeCount')
        vals = M.getAttr('x', z)
        selected = gp.tuplelist((i, j)
                                for i, j in vals.keys() if vals[i, j] > 0.5)

        path = af.subtour(selected)

        path_P = []

        for p in path:
            comp = data[p]
            path_P.append([x[(p, 0, 0)].X, x[(p, 0, 1)].X])
            path_P.append([x[(p, 1, 0)].X, x[(p, 1, 1)].X])

        print('Ruta: \n' + str(path))

        # distancia = 0
        # for c in range(m - 1):
        #     distancia += np.linalg.norm(
        #         np.array(path_P[c]) - np.array(path_P[c + 1]))
        #
        # distancia += np.linalg.norm(
        #     np.array(path_P[m - 1]) - np.array(path_P[0]))

        ax2 = plt.gca()

        # Puntos de entrada y salida
        circulos = []
        for p in path_P:
            circulo = Circle((p[0], p[1]), 0.3, color='black', alpha=1)
            circulos.append(circulo)
            ax2.add_patch(circulo)

        polygon = Polygon(path_P, fill=False, linestyle='-', alpha=1)
        ax2.add_patch(polygon)

        plt.title(str(m) + "- XPPN - Mode " + str(modo) + 'No tour')
        string = 'imagenes/' + \
            str(m) + "-XPPN - DFJ - Mode " + str(modo) + "No tour.png"
        plt.savefig(string)
        plt.close()
        M.write('error.lp')
        print('No se ha resuelto')

        resultados.append(end_time - start_time)
        resultados.append(gap)
        resultados.append(nodeCount)

        # if datos.init:
        #     resultados.append(np.nan)
        #     resultados.append(obj_heur)
        #     resultados.append(res_time_heur)

        return resultados

    obj_model = M.ObjVal
    gap = M.getAttr('MIPGap')
    nodeCount = M.getAttr('NodeCount')

    # resultados.append(end_time - start_time)
    # resultados.append(gap)
    # resultados.append(nodeCount)
    # resultados.append(obj_model)

    # if datos.init:
    #     resultados.append(obj_heur)
    #     resultados.append(res_time_heur)
    # else:
    #     resultados.append(np.nan)
    #     resultados.append(np.nan)

    # resultados.append(tour)
    if (datos.show):

        path_P = []

        vals_landa = M.getAttr('x', landa)
        vals_landa


        for p in tour:
            comp = data[p]
            if type(comp) is Poligonal:
                if u[p].X < 0.5:
                    path_P.append([x[p, 0, 0].X, x[p, 0, 1].X])
                    for i in np.arange(0, comp.num_puntos):
                        if i <= round(vals_landa[(p, 0)], 3) and i >= round(vals_landa[(p, 1)], 3):
                            print(i)
                            path_P.append(comp.V[i])
                    path_P.append([x[p, 1, 0].X, x[p, 1, 1].X])
                else:
                    path_P.append([x[(p, 0, 0)].X, x[(p, 0, 1)].X])
                    for i in range(comp.num_puntos):
                        if i >= round(vals_landa[(p, 0)],3) and i <= round(vals_landa[(p, 1)], 3):
                            print(i)
                            path_P.append(comp.V[i-1])
                    path_P.append([x[(p, 1, 0)].X, x[(p, 1, 1)].X])
            else:
                path_P.append([x[p, 0, 0].X, x[p, 0, 1].X])
                # path_P.append([x[(p, 1, 0)].X, x[(p, 1, 1)].X])

        print('Ruta: \n' + str(tour))
        
        # print(path_P)
        #
        # distancia = 0
        # for c in range(m - 1):
        #     distancia += np.linalg.norm(
        #         np.array(path_P[c]) - np.array(path_P[c + 1]))
        #
        # distancia += np.linalg.norm(
        #     np.array(path_P[m - 1]) - np.array(path_P[0]))
    
    
    def junta_listas(listas, i, j):
        # Tenemos una lista de listas y quiero juntar la lista i y la lista j y colocarla donde esta la lista i
        listas[i] += listas[j]
        del(listas[j])
        return listas
    
    def deshaz_listas(listas, i):
        lista = listas[i]
        listas[i] = lista[0:-1]
        listas.insert(i+1, [lista[-1]])
        return listas
        
    # lista = [[1], [2], [3], [4]]
    #
    # junta_listas(lista, 1, 2)
    # junta_listas(lista, 1, 2)
    #
    # print(lista)
    #
    # deshaz_listas(lista, 1)
    #
    # print(lista)
        
        
        
    best_obj = 1e7
    best_list = []
    
    lista_init = []

    for t in tour:
        lista_init.append([t])
    
    # count = 0
    # tour = [0, 4, 3, 5, 1, 2]
    for count in range(len(tour)-1):
        solution_list = []

        for t in tour:
            solution_list.append([t])
                
        while count+1 < len(solution_list):
            
            # for j in range(i+1, len(tour)):
            while(1): 
                junta_listas(solution_list, count, count+1)
                
                # Tengo la lista [[0, 4], [3], [5], [1], [2]]
                print('La lista que he conseguido es la siguiente: ' + str(solution_list))
                print(count)

                # Resuelvo el problema
                objetivo = MultiTargetPolygonal(datos, solution_list)
                
                # Si no encuentro solucion inicial
                if objetivo < 0:
                    deshaz_listas(solution_list, count)
                    print('Rompo la lista y obtengo: ' + str(solution_list))
                    break
                    
                
                if objetivo <= best_obj and objetivo >= 0:
                    best_obj = objetivo
                    print('Actualizo lista: ' + str(solution_list))
                    best_list = solution_list
                    
                
                if count+1 == len(solution_list):
                    break
                
            count += 1
                
        
    print(best_obj)
    print(best_list)
    
    
    
    # print('\nDistancia total (MTZ): ' + str(obj_model))
    # print('Tiempo de resolución: ' + str(end_time - start_time))
    # print('Gap restante: ' + str(gap))
    # print('Número de nodos visitados: ' + str(nodeCount))
    
    
    end_time = time.time()

    heur_time = end_time - start_time
    
    return best_list, heur_time


def MultiTargetPolygonal(datos, listas):
    vD = datos.vD
    vC = datos.vC
    
    n = 2
    nG = datos.m

    
    data = datos.mostrar_datos()
    global sec
    sec = 0
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if model.cbGet(GRB.Callback.MIPSOL_SOLCNT) == 0:
                # creates new model attribute '_startobjval'
                model._startobjval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            
    # nE = 2
    # np.random.seed(15) 338.90
    # datos.orig = np.random.uniform(0, 100, 2).tolist()
    
    # datos.orig = [50, 50]
    
    # datos.dest = datos.orig
    #
    # datos1 = Data([], nE, 3, 1, None, False, True, 0)
    # datos1.generar_muestra()
    # E = datos1.data
    
    # E_index = range(nE)
    T_index = range(datos.m+2)
    T_index_prima = range(1, datos.m+1)
    T_index_primaprima = range(0, datos.m+1)
    N = range(n)
    
    # Creacion del modelo
    MODEL = gp.Model('klmedian-continuous')
    
    
    # Variable binaria ugt = 1 si se entra por el grafo g en la etapa t
    
    ugt_index = []
    
    for g in T_index_prima:
        for t in T_index_prima:
            ugt_index.append((g, t))
    
    ugt = MODEL.addVars(ugt_index, vtype=GRB.BINARY, name='ugt')
    
    
    # Variable binaria ugt = 1 si se entra por el grafo g en la etapa t
    
    mugt_index = ugt_index
    
    mugt = MODEL.addVars(mugt_index, vtype=GRB.BINARY, name='mugt')
    kt = MODEL.addVars(T_index_prima, vtype=GRB.CONTINUOUS, name='kt')
    

    pgt = MODEL.addVars(mugt_index, vtype = GRB.CONTINUOUS, name = 'pgt')
    
    # Variable continua no negativa dgLt que indica la distancia desde el punto de lanzamiento hasta el grafo g.
    
    dgLt_index = ugt_index
    
    dgLt = MODEL.addVars(dgLt_index, vtype=GRB.CONTINUOUS, lb=0.0, name='dgLt')
    difgLt = MODEL.addVars(dgLt_index, 2, vtype=GRB.CONTINUOUS, lb=0.0, name='difgLt')
    
    # Variable continua no negativa pgLit = ugit * dgLit
    pgLt_index = ugt_index
    
    pgLt = MODEL.addVars(pgLt_index, vtype=GRB.CONTINUOUS, lb=0.0, name='pgLt')
    
    # Variable binaria yggt = 1 si voy de g1 a g2.
    yggt_index = []
    
    
    for g1 in T_index_prima:
        for g2 in T_index_prima:
            if g1 != g2:
                for t in T_index_prima:
                    yggt_index.append((g1, g2, t))
    
    yggt = MODEL.addVars(yggt_index, vtype = GRB.BINARY, name = 'yggt')
    dggt = MODEL.addVars(yggt_index, vtype = GRB.CONTINUOUS, name = 'Dggt')
    difggt = MODEL.addVars(yggt_index, 2, vtype = GRB.CONTINUOUS, name = 'difggt')
    pggt = MODEL.addVars(yggt_index, vtype = GRB.CONTINUOUS, name = 'pggt')
    
    
    # Variable continua sgt que indica el orden en la etapa
    sgt_index = []
    
    for t in T_index_prima:
        for g in T_index_prima:
            sgt_index.append((g, t))
    
    sgt = MODEL.addVars(sgt_index, vtype = GRB.CONTINUOUS)
    
    # Variable binaria vgt = 1 si en la etapa t salimos por el grafo g
    vgt_index = ugt_index
    
    vgt = MODEL.addVars(vgt_index, vtype=GRB.BINARY, name='vgt')
    
    # Variable continua no negativa dgRit que indica la distancia desde el punto de salida del segmento sgi hasta el
    # punto de recogida del camion
    dgRt_index = ugt_index
    
    dgRt = MODEL.addVars(dgRt_index, vtype=GRB.CONTINUOUS, lb=0.0, name='dgRt')
    difgRt = MODEL.addVars(dgRt_index, 2, vtype=GRB.CONTINUOUS, lb=0.0, name='difgRt')
    
    # Variable continua no negativa pgRt = ugt * dgRt
    pgRt_index = ugt_index
    
    pgRt = MODEL.addVars(pgRt_index, vtype=GRB.CONTINUOUS, lb=0.0, name='pgRt')
    
    # Variable continua no negativa dRLt que indica la distancia entre el punto de recogida en la etapa t ugt el punto de
    # salida para la etapa t+1
    dRLt_index = T_index_primaprima
    
    dRLt = MODEL.addVars(dRLt_index, vtype=GRB.CONTINUOUS, lb=0.0, name='dRLt')
    difRLt = MODEL.addVars(dRLt_index, 2, vtype=GRB.CONTINUOUS, lb=0.0, name='difRLt')
    
    # Variable continua no negativa dLRt que indica la distancia que recorre el camión en la etapa t mientras el dron se mueve
    dLRt_index = T_index
    
    dLRt = MODEL.addVars(dLRt_index, vtype=GRB.CONTINUOUS, lb=0.0, name='dLRt')
    difLRt = MODEL.addVars(dLRt_index, 2, vtype=GRB.CONTINUOUS, lb=0.0, name='difLRt')
    
    # Variables que modelan los puntos de entrada o recogida
    # xLt: punto de salida del dron del camion en la etapa t
    xLt_index = []
    
    for t in T_index:
        for dim in range(2):
            xLt_index.append((t, dim))
    
    xLt = MODEL.addVars(xLt_index, vtype=GRB.CONTINUOUS, name='xLt')
    
    # xRt: punto de recogida del dron del camion en la etapa t
    xRt_index = []
    
    for t in T_index:
        for dim in range(2):
            xRt_index.append((t, dim))
    
    xRt = MODEL.addVars(xRt_index, vtype=GRB.CONTINUOUS, name='xRt')
    
    # Generando los mus de la envolvente convexa, los landas de la poligonal y las
    # variables binarias que indican qué segmento se elige
    
    landa_index = []
    rho_index = []
    gammaR_index = []
    gammaL_index = []
    muR_index = []
    muL_index = []
    u_index = []
    Rp_index = []
    Lp_index = []
    
    for p in T_index_prima:
        comp = data[p-1]
        # if type(comp) is Poligono:
        #     for mu in range(comp.num_puntos):
        #         mu_index.append((c, mu))
        if type(comp) is Poligonal:
            u_index.append(p)
            # landa de la variable de entrada en la poligonal c
            rho_index.append(p)
            # landa de la variable de salida en la poligonal c
            landa_index.append(p)
            for segm in range(comp.num_segmentos):
                muR_index.append((p, segm))
                muL_index.append((p, segm))
            for punto in range(comp.num_puntos):
                gammaR_index.append((p, punto))
                gammaL_index.append((p, punto))
        for dim in N:
            Rp_index.append((p, dim))
            Lp_index.append((p, dim))
    
    
    landa = MODEL.addVars(landa_index, vtype=GRB.CONTINUOUS, name='landa')
    rho = MODEL.addVars(rho_index, vtype=GRB.CONTINUOUS, name='rho')
    
    gammaR = MODEL.addVars(gammaR_index, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='gammaR')
    gammaL = MODEL.addVars(gammaL_index, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='gammaL')
    
    muR = MODEL.addVars(muR_index, vtype=GRB.BINARY, name='muR')
    muL = MODEL.addVars(muL_index, vtype=GRB.BINARY, name='muL')
        
    u = MODEL.addVars(u_index, vtype=GRB.BINARY, name='u')
    dt = MODEL.addVars(T_index_prima, vtype = GRB.CONTINUOUS, name = 'dt')
    
    landa_min = MODEL.addVars(u_index, vtype=GRB.CONTINUOUS, lb=0.0, name='landa_min')
    landa_max = MODEL.addVars(u_index, vtype=GRB.CONTINUOUS, lb=0.0, name='landa_max')
    
    Rp = MODEL.addVars(Rp_index, vtype = GRB.CONTINUOUS, name = 'Rp')
    Lp = MODEL.addVars(Lp_index, vtype = GRB.CONTINUOUS, name = 'Lp')
    
    beta = MODEL.addVars(T_index_prima, vtype = GRB.BINARY, name = 'beta')
    
    MODEL.update()
    
    # mugt_inicial, heur_time = heuristic(datos)

    # for lista, t in zip(listas, range(1, len(listas)+1)):
    #
    #     if len(lista) == 1:
    #         mugt[lista[0]+1, t].start = 1
    #         ugt[lista[0]+1, t].start = 1
    #         vgt[lista[0]+1, t].start = 1
    #
    #
    #     if len(lista) > 1:
    #
    #         mugt[lista[0]+1, t].start = 1
    #         ugt[lista[0]+1, t].start = 1
    #
    #         for i in range(1, len(lista)-1):
    #             mugt[lista[i]+1, t].start = 1
    #             yggt[lista[i]+1, lista[i+1]+1, t].start = 1
    #
    #         mugt[lista[-1]+1, t].start = 1
    #         vgt[lista[-1]+1, t].start = 1            
    
    for lista, t in zip(listas, range(1, len(listas)+1)):

        if len(lista) == 1:
            MODEL.addConstr(mugt[lista[0]+1, t] >= 0.5)
            MODEL.addConstr(ugt[lista[0]+1, t] >= 0.5)
            MODEL.addConstr(vgt[lista[0]+1, t] >= 0.5)
            
            MODEL.addConstrs(yggt[g1, g2, t] <= 0.5 for g1 in T_index_prima for g2 in T_index_prima if g1 != g2)
            
        
        if len(lista) > 1:
            
            
            MODEL.addConstr(mugt[lista[0]+1, t] >= 0.5)
            MODEL.addConstr(ugt[lista[0]+1, t] >= 0.5)
            
            for i in range(1, len(lista)-1):
                MODEL.addConstr(mugt[lista[i]+1, t] >= 0.5)
                MODEL.addConstr(yggt[lista[i]+1, lista[i+1]+1, t] >= 0.5)
                
            MODEL.addConstr(mugt[lista[-1]+1, t] >= 0.5)
            MODEL.addConstr(vgt[lista[-1]+1, t] >= 0.5)
                
        #
        # for g, t in mugt.keys():
        #     if (g, t) in mugt_start:
        #         mugt[g, t].start = 1
        #     else:
        #         mugt[g, t].start = 0
        #

        # MODEL.read('./sol_files/multitarget.sol')
        
        
    # (1), (2): for each operation the drone only can enter and exit, respectively,by one target
    MODEL.addConstrs(ugt.sum('*', t) <= 1 for t in T_index_prima)
    MODEL.addConstrs(vgt.sum('*', t) <= 1 for t in T_index_prima)
    
    # (3): every target will be visited in some operation
    MODEL.addConstrs(mugt.sum(g, '*') == 1 for g in T_index_prima)
    
    # (4): if target t is visited by the drone for the operation o, one of two alternative situations must occur: either t is the first target 
    # for the operation o or target t is visited by the drone after visiting another target t' for the operation o. 
        
    MODEL.addConstrs(mugt[g, t] - ugt[g, t] == yggt.sum('*', g, t) for g, t in ugt.keys())
    
    # (5): state that if the target t for the operation o is visited by the drone, either t is the last target of the operation, 
    # or the drone must move to another target t' of the operation o after visiting target t.
    MODEL.addConstrs(mugt[g, t] - vgt[g, t] == yggt.sum(g, '*', t) for g, t in ugt.keys())
    
    
    # (6): Subtour elimination constraints
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))
    
    # for t in T_index_prima:
    #     restricciones = MODEL.addConstrs(gp.quicksum(yggt[g1, g2, t] for g1, g2 in permutations(set, 2)) <= len(set) - 1 for set in list(powerset(T_index_prima)))
    #     restricciones.Lazy = 2
    
    # Monotonicity
    for t in T_index_prima[0:-1]:
        MODEL.addConstr(beta[t] <= beta[t+1])
        
    # Definition of kt
    for t in T_index_prima:
        kt[t] = mugt.sum('*', t)
  
    
    # (VI-1)
    for t in T_index_prima:
        MODEL.addConstrs(gp.quicksum(kt[t1] for t1 in T_index_prima if t1 < t) >= datos.m*beta[t] for t in T_index_prima)
    

    # (VI-2)
    for t in T_index_prima:
        MODEL.addConstr(kt[t] >= 1 - beta[t])
    
    
    MODEL.addConstrs(difggt[g1, g2, t, dim] >=  Lp[g1, dim] - Rp[g2, dim] for g1, g2, t, dim in difggt.keys())
    MODEL.addConstrs(difggt[g1, g2, t, dim] >= -Lp[g1, dim] + Rp[g2, dim] for g1, g2, t, dim in difggt.keys())
    MODEL.addConstrs(gp.quicksum(difggt[g1, g2, t, dim]*difggt[g1, g2, t, dim] for dim in N) <= dggt[g1, g2, t] * dggt[g1, g2, t] for g1, g2, t in dggt.keys())
    
    # BigM = 10000
    # SmallM = 0
    
    # MODEL.addConstr(beta[2] == 0)
    
    for g1, g2, t in yggt.keys():
    
        comp1 = data[g1-1]
        comp2 = data[g2-1]
    
        BigM = eM.estima_BigM_local(comp1, comp2)
        SmallM = eM.estima_SmallM_local(comp1, comp2)
    
        MODEL.addConstr(pggt[g1, g2, t] >= SmallM * yggt[g1, g2, t])
        MODEL.addConstr(pggt[g1, g2, t] >= dggt[g1, g2, t] - BigM * (1 - yggt[g1, g2, t]))
    
    
    MODEL.addConstrs((difgLt[g, t, dim] >=  xLt[t, dim] - Rp[g, dim]) for g, t, dim in difgLt.keys())
    MODEL.addConstrs((difgLt[g, t, dim] >= -xLt[t, dim] + Rp[g, dim]) for g, t, dim in difgLt.keys())
    
    MODEL.addConstrs(difgLt[g, t, 0]*difgLt[g, t, 0] + difgLt[g, t, 1]*difgLt[g, t, 1] <= dgLt[g, t]*dgLt[g, t] for g, t in dgLt.keys())
    
    SmallM = 0
    
    BigM = max([np.linalg.norm(np.array(p) - np.array(q)) for comp1 in data for p in comp1.V for comp2 in data for q in comp2.V])
    # BigM = 10000
    
    MODEL.addConstrs(pgLt[g, t] >= SmallM * ugt[g, t] for g, t in pgLt.keys())
    MODEL.addConstrs(pgLt[g, t] >= dgLt[g, t] - BigM * (1 - ugt[g, t]) for g, t in pgLt.keys())
    
    
    MODEL.addConstrs((difgRt[g, t, dim] >=   Lp[g, dim] - xRt[t, dim]) for g, t, dim in difgRt.keys())
    MODEL.addConstrs((difgRt[g, t, dim] >=  -Lp[g, dim] + xRt[t, dim]) for g, t, dim in difgRt.keys())
    
    MODEL.addConstrs(difgRt[g, t, 0]*difgRt[g, t, 0] + difgRt[g, t, 1]*difgRt[g, t, 1] <= dgRt[g, t]*dgRt[g, t] for g, t in dgRt.keys())
    
    MODEL.addConstrs(pgRt[g, t] >= SmallM * vgt[g, t] for g, t in pgRt.keys())
    MODEL.addConstrs(pgRt[g, t] >= dgRt[g, t] - BigM * (1 - vgt[g, t]) for g, t in pgRt.keys())
    
    MODEL.addConstrs((difRLt[t, dim] >=   xRt[t, dim] - xLt[t + 1, dim]) for t in T_index_primaprima for dim in range(2))
    MODEL.addConstrs((difRLt[t, dim] >=  -xRt[t, dim] + xLt[t + 1, dim]) for t in T_index_primaprima for dim in range(2))
    MODEL.addConstrs(difRLt[t, 0]*difRLt[t, 0] + difRLt[t, 1] * difRLt[t, 1] <= dRLt[t] * dRLt[t] for t in T_index_primaprima)
    
    MODEL.addConstrs((difLRt[t, dim] >=   xLt[t, dim] - xRt[t, dim]) for t, dim in difLRt.keys())
    MODEL.addConstrs((difLRt[t, dim] >= - xLt[t, dim] + xRt[t, dim]) for t, dim in difLRt.keys())
    MODEL.addConstrs(difLRt[t, 0]*difLRt[t, 0] + difLRt[t, 1] * difLRt[t, 1] <= dLRt[t] * dLRt[t] for t in dLRt.keys())
    
    
    for g, t in mugt.keys():
        comp = data[g-1]
        if type(comp) is Poligonal:
            SmallM = comp.alpha*comp.longitud
            BigM = comp.longitud
            MODEL.addConstr(pgt[g, t] >= SmallM * mugt[g, t])
            MODEL.addConstr(pgt[g, t] >= dt[g] - BigM* (1 - mugt[g, t]))
        if type(comp) is Punto:
            SmallM = 0
            BigM = 0
            MODEL.addConstr(pgt[g, t] >= SmallM * mugt[g, t])
            MODEL.addConstr(pgt[g, t] >= dt[g] - BigM* (1 - mugt[g, t]))
    
    MODEL.addConstrs((pgLt.sum('*', t) + pgt.sum('*', t) + gp.quicksum(pggt[g1, g2, t] for g1 in T_index_prima for g2 in T_index_prima if g1 != g2) + pgRt.sum('*', t))/vD <= dLRt[t]/vC for t in T_index_prima)
    #
    # MODEL.addConstrs((gp.quicksum(pgLit[g, i, t] for i in grafos[g-1].aristas) + pgij.sum(g, '*', '*') +  gp.quicksum(pgi[g, i]*grafos[g-1].longaristas[i // 100 - 1, i % 100] for i in grafos[g-1].aristas) + gp.quicksum(pgRit[g, i, t] for i in grafos[g-1].aristas))/vD <= dLRt[t]/vC + BigM*(1- gp.quicksum(ugit[g, i, t] for i in grafos[g-1].aristas)) for t in T_index_prima for g in T_index_prima)

    # MODEL.addConstrs((pgLt[g, t] + pgt[g, t] + pggt.sum('*', '*', t) + pgRt[g, t])/vD <= dLRt[t]/vC + BigM*(1 - mugt[g, t]) for g in T_index_prima for t in T_index_prima)
    
    MODEL.addConstrs((dLRt[t]/vC <= datos.capacity) for t in dLRt.keys())
    
    for p in T_index_prima:
        comp = data[p-1]
        if type(comp) is Poligonal:
            MODEL.addConstr(rho[p] - landa[p] == landa_max[p] - landa_min[p], name='u0')
            # si u = 0, entonces landa0 >= landa1
            MODEL.addConstr(landa_max[p] + landa_min[p] >= comp.alpha * comp.num_segmentos, name='u1')
            MODEL.addConstr(landa_max[p] <= comp.num_segmentos * (1 - u[p]), name='u2')
            MODEL.addConstr(landa_min[p] <= comp.num_segmentos * u[p], name='u3')
            MODEL.addConstr(dt[p] == (landa_max[p] + landa_min[p])*comp.longitud/comp.num_segmentos)
    
            print(comp.alpha*comp.longitud)
            MODEL.addConstrs(rho[p] - punto >= gammaR[p, punto] - comp.num_puntos*(1 - muR[p, punto-1]) for punto in range(1, comp.num_puntos))
            MODEL.addConstrs(rho[p] - punto <= gammaR[p, punto] + comp.num_puntos*(1 - muR[p, punto-1]) for punto in range(1, comp.num_puntos))
            MODEL.addConstr(gammaR[p, 0] <= muR[p, 0])
            MODEL.addConstrs(gammaR[p, punto] <= muR[p, punto-1] + muR[p, punto] for punto in range(1, comp.num_segmentos))
            MODEL.addConstr(gammaR[p, comp.num_segmentos] <= muR[p, comp.num_segmentos-1])
            MODEL.addConstr(muR.sum(p, '*') == 1)
            MODEL.addConstr(gammaR.sum(p, '*') == 1)
            MODEL.addConstrs(Rp[p, dim] == gp.quicksum(gammaR[p, punto]*comp.V[punto][dim] for punto in range(comp.num_puntos)) for dim in N)
    
            MODEL.addConstrs(landa[p] - punto >= gammaL[p, punto] - comp.num_puntos*(1 - muL[p, punto-1]) for punto in range(1, comp.num_puntos))
            MODEL.addConstrs(landa[p] - punto <= gammaL[p, punto] + comp.num_puntos*(1 - muL[p, punto-1]) for punto in range(1, comp.num_puntos))
            MODEL.addConstr(gammaL[p, 0] <= muL[p, 0])
            MODEL.addConstrs(gammaL[p, punto] <= muL[p, punto-1] + muL[p, punto] for punto in range(1, comp.num_segmentos))
            MODEL.addConstr(gammaL[p, comp.num_segmentos] <= muL[p, comp.num_segmentos-1])
            MODEL.addConstr(muL.sum(p, '*') == 1)
            MODEL.addConstr(gammaL.sum(p, '*') == 1)
            MODEL.addConstrs(Lp[p, dim] == gp.quicksum(gammaL[p, punto]*comp.V[punto][dim] for punto in range(comp.num_puntos)) for dim in N)
        if type(comp) is Punto:
            for dim in N:
                MODEL.addConstr(Lp[p, dim] == comp.V[dim])
                MODEL.addConstr(Rp[p, dim] == comp.V[dim])
                MODEL.addConstr(dt[p] == 0)
            
    MODEL.addConstrs(xLt[0, dim] == datos.orig[dim] for dim in N)
    MODEL.addConstrs(xRt[0, dim] == datos.orig[dim] for dim in N)
    #
    MODEL.addConstrs(xLt[datos.m+1, dim] == datos.dest[dim] for dim in N)
    MODEL.addConstrs(xRt[datos.m+1, dim] == datos.dest[dim] for dim in N)

    MODEL.update()
    
    # Funcion objetivo
    # + gp.quicksum(0.5*pgLt[index] for index in pgLt.keys()) + gp.quicksum(0.5*pgRt[index] for index in pgRt.keys())
    # objective = gp.quicksum(dRLt[index] for index in dRLt.keys()) + gp.quicksum(dLRt[index] for index in dLRt.keys()) + gp.quicksum(pgLt[index] for index in dgLt.keys()) + gp.quicksum(pgRt[index] for index in dgRt.keys())
    
    objective = gp.quicksum(3*dLRt[index] for index in dLRt.keys()) + gp.quicksum(3*dRLt[index] for index in dRLt.keys()) + gp.quicksum(pgRt[index] for index in pgRt.keys()) + gp.quicksum(pgLt[index] for index in pgLt.keys()) + gp.quicksum(pggt[g1, g2, t] for g1, g2, t in yggt.keys()) + gp.quicksum(pgt[index] for index in pgt.keys())
    
    MODEL.setObjective(objective, GRB.MINIMIZE)
    
    MODEL.update()
    
    # MODEL.setParam('TimeLimit', datos.tmax)
    
    MODEL.Params.Threads = 5
    # MODEL.Params.SolutionLimit = 1
    # MODEL.Params.TimeLimit = 1
    MODEL.Params.LazyConstraints = 1
    MODEL.Params.OutputFlag = 1
    # MODEL.Params.FeasibilityTol = 1e-3
    
    MODEL.update()
    # MODEL.Params.SolutionLimit = 3
    
    # Optimizamos
    MODEL.optimize()
    # MODEL.optimize(callback)



    # MODEL.optimize()
    # MODEL.optimize(subtourelim)
    
    # MODEL.write('aver.lp')

    
    if MODEL.Status == 3 or MODEL.SolCount == 0:
        
        # MODEL.computeIIS()
        # MODEL.write('infactible.ilp')
        
        return -1
    
    # polygon_camion = Polygon(path_camion, fill=False, linestyle=':', alpha=1, color='red')
    # ax.add_patch(polygon_camion)
    
    
    # plt.title(str(datos.m) + "coordinate ")
    # string = 'imagenes/' + str(m) + "-XPPN - MTZ - Mode " + \
    #     str(modo) + " - Radii - " + str(datos.r) + ".png"
    # plt.savefig('Instance{b}-{d}.png'.format(b = datos.m, d = datos.capacity))
    
    # plt.show()
    return MODEL.ObjVal
    # return MODEL._startobjval  

# MTZ2(datos)
# objetivos_init = []
# objetivos_mtz = []
# objetivos_orig = []
#
# # seed 38
# # x = np.linspace(0, 1, 21)
# x = range(2)
# for i in x:
#     np.random.seed(i)
#     datos = Data([], m=10,
#                  r=3,
#                  modo=1,
#                  tmax=150,
#                  elim=0,
#                  init = 0,
#                  pol = 1,
#                  show=True,
#                  seed=2)
#     datos.generar_muestra()
#     resultados = MTZ(datos)
#     # resultados = approach_MTZ(datos)
#     objetivos_init.append(resultados[0])
#
#
#     np.random.seed(i)
#     datos = Data([], m=10,
#                  r=3,
#                  modo=1,
#                  tmax=30,
#                  elim=1,
#                  init = 0,
#                  pol = 1,
#                  show=True,
#                  seed=2)
#     # datos.reduce_radio(i)
#     datos.generar_muestra()
#     resultados = MTZ(datos)
#     objetivos_mtz.append(resultados[0])
#
#
# objetivos_init
# objetivos_mtz
# fig, ax = plt.subplots()
# ax.boxplot(objetivos_init)
# ax.boxplot(objetivos_mtz)
# plt.show()
# print(objetivos_init)
# print(objetivos_orig)
# print(x)
