#!/usr/bin/python

# Copyright 2019, Gurobi Optimization, LLC

# Solve the classic diet model, showing how to add constraints
# to an existing model.


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
# Con 15 todavia no la encuentra
# np.random.seed(112)
# np.random.seed(11112)
# 168.60477938110762
# np.random.seed(5)
#
# datos = Data([], m = 10,
#                  r = 2,
#                  modo = 4,
#                  tmax = 1200,
#                  init = True,
#                  prepro = 0,
#                  refor = 0,
#                  show = True,
#                  seed = 2)
# datos.generar_muestra()
# #
# prueba = 1
def MTZ(datos):

    data = datos.mostrar_datos()
    
    data.insert(0, Punto(datos.orig))
    
    data.append(Punto(datos.dest))

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
    
    M.addConstr(sc[m-1] == m-1, name = 'rest23')

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

    end_time = time.time()

    
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

    resultados.append(end_time - start_time)
    resultados.append(gap)
    resultados.append(nodeCount)
    resultados.append(obj_model)

    # if datos.init:
    #     resultados.append(obj_heur)
    #     resultados.append(res_time_heur)
    # else:
    #     resultados.append(np.nan)
    #     resultados.append(np.nan)

    resultados.append(tour)

    
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
        
        print(path_P)

        distancia = 0
        for c in range(m - 1):
            distancia += np.linalg.norm(
                np.array(path_P[c]) - np.array(path_P[c + 1]))

        distancia += np.linalg.norm(
            np.array(path_P[m - 1]) - np.array(path_P[0]))
        
        
    data.pop(0)
    data.pop(-1)
        
        # print(data)
        # if not(datos.init):
        #     fig = datos.dibujar_muestra()
        #     ax2 = fig.add_subplot(111)
        # else:
        #     ax2 = plt.gca()

        # Puntos de entrada y salida
        # circulos = []
        # for p in path_P:
        #     circulo = Circle((p[0], p[1]), 0.3, color='black', alpha=1)
        #     circulos.append(circulo)
        #     ax2.add_patch(circulo)
        #
        # polygon = Polygon(path_P, fill=False, linestyle='-', alpha=1, lw = 0.3)
        # ax2.add_patch(polygon)

        # if datos.pol:
        #     polygon2 = Polygon(poligono, fill = False, linestyle = 'dotted', alpha = 1, color = 'red')
        #     ax2.add_patch(polygon2)


        # plt.title(str(m) + "- XPPN - Mode " + str(modo))
        # string = './imagenes/' + \
        #     str(m) + "-XPPN - MTZ - Mode " + \
        #     str(modo) + " - Radii - " + str(datos.r)
        # if datos.init:
        #     string += ' init '
        # else:
        #     string += ' real '
        # if datos.prepro:
        #     string += ' prepro '
        # if datos.elim:
        #     string += ' elim '
        # if datos.refor:
        #     string += ' refor '
        # if datos.pol:
        #     string += ' pol'
        # string += '.png'
        # plt.savefig(string)
        # plt.show(block=False)
        # plt.pause(2)
        # # plt.show()
        # plt.close()
        # plt.clf(fig2)

    # if datos.init:
        # print('\nDistancia total (Heuristic): ' + str(obj_heur))
        # print('Tiempo de resolución (Heuristic): ' + str(res_time_heur))

    print('\nDistancia total (MTZ): ' + str(obj_model))
    print('Tiempo de resolución: ' + str(end_time - start_time))
    print('Gap restante: ' + str(gap))
    print('Número de nodos visitados: ' + str(nodeCount))
    
    heur_time = end_time - start_time
    
    return tour, heur_time


# MTZ(datos)
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
