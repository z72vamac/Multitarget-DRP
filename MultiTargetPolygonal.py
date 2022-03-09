"""Tenemos un conjunto E de entornos ugt un conjunto de poligonales P de las que queremos recorrer un porcentaje alfa p . Buscamos
   un tour de mínima distancia que alterne poligonal-entorno ugt que visite todas las poligonales"""
   
" Esto permite preprocesado de buscar conjuntos de variables ys que no pueden ser uno simultaneamente"

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
from auxiliar_functions import path2matrix, min_dist
from preprocessing3 import preprocessing3
import auxiliar_functions as af
from heuristic import heuristic
from MTZ import MTZ
from MTZ2 import MTZ2
from MTZ3 import MTZ3

# Definicion de los datos
""" P: conjunto de poligonales a agrupar
    E: conjunto de entornos
    T: sucesion de etapas
    C(e): centro del entorno e
    R(e): radio del entorno e
    p: indice de las poligonales
    e: indice de los entornos
    t: indice de las etapas
    n: dimension del problema
"""

# np.random.seed(1)
#
# m = 10

# datos = Data([], m = 13,
#                  r = 4,
#                  capacity = 40,
#                  modo = 2,
#                  tmax = 1800,
#                  init = 0,
#                  show = True,
#                  seed = 2)
# datos.generar_muestra()
#
# data = datos.mostrar_datos()

# n = 2
# datos.orig = [0, 0]
# datos.dest = [0, 0]


def MultiTargetPolygonal(datos, log):
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
                model._starttime = model.cbGet(GRB.Callback.RUNTIME)
                model.terminate()
                
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
        
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            ys = model.cbGetSolution(model._yggt)
            us = model.cbGetSolution(model._ugt)
            vs = model.cbGetSolution(model._vgt)
            
            # mus = model.cbGetSolution(model._mugt)

            for t in T_index_prima:
                selected_ys = gp.tuplelist((g1, g2) for g1, g2, t1 in model._yggt.keys() if ys[g1, g2, t] > 0.5 and t1 == t)
                selected_us = gp.tuplelist((g1, t1) for g1, t1 in model._ugt.keys() if us[g1, t] > 0.5 and t1 == t)
                selected_vs = gp.tuplelist((g1, t1) for g1, t1 in model._vgt.keys() if vs[g1, t] > 0.5 and t1 == t)
                

                
                if len(selected_ys) > 0:
                    # print("ugt = " + str(list(selected_us)))
                    print("yggt = " + str(list(selected_ys)))
                    # print("vgt = " + str(list(selected_vs))) 
                    
                    
                    tour = subtour(selected_ys)
                    
                    # print(tour)
                    
                    # selected_mus = gp.tuplelist((g, t) for g, t1 in model._mugt.keys() if mus[g, t] > 0.5 and t1 == t)
                    # print(selected_mus)
    
                    if len(tour) > 1:
                        print('tour = ' + str(tour))
    
                        for t in T_index_prima:
                            model.cbLazy(gp.quicksum(model._yggt[g1, g2, t] for g1, g2 in permutations(tour, 2)) <= len(tour) - 1.00001)
                            print(gp.quicksum(model._yggt[g1, g2, t] for g1, g2 in permutations(tour, 2)))
                            global sec
                            sec += 1
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
    zsegm_index = []
    
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
                for segm2 in range(comp.num_segmentos):
                    zsegm_index.append((p, segm, segm2))
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
    
    zsegm1segm2 = MODEL.addVars(zsegm_index, vtype = GRB.BINARY, name = 'zsegm1segm2')
    dsegm1segm2 = MODEL.addVars(zsegm_index, vtype = GRB.CONTINUOUS, lb =0.0, name = 'dsegm1segm2')
    psegm1segm2 = MODEL.addVars(zsegm_index, vtype = GRB.CONTINUOUS, lb =0.0, name = 'psegm1segm2')
    
    
    landa_min = MODEL.addVars(u_index, vtype=GRB.CONTINUOUS, lb=0.0, name='landa_min')
    landa_max = MODEL.addVars(u_index, vtype=GRB.CONTINUOUS, lb=0.0, name='landa_max')
    
    Rp = MODEL.addVars(Rp_index, vtype = GRB.CONTINUOUS, name = 'Rp')
    Lp = MODEL.addVars(Lp_index, vtype = GRB.CONTINUOUS, name = 'Lp')
    
    beta = MODEL.addVars(T_index_prima, vtype = GRB.BINARY, name = 'beta')
    
    MODEL.update()
    
    # mugt_inicial, heur_time = heuristic(datos)

    if datos.init == 1:
    
        tour, heuristic_time = MTZ(datos)
        
        for t in T_index_prima:
            mugt[tour[t-1]+1, t].start = 1
            ugt[tour[t-1]+1, t].start = 1
            vgt[tour[t-1]+1, t].start = 1
            
        for g1, g2, t in yggt.keys():
            yggt[g1, g2, t].start = 0
    
    if datos.init == 2:
        
        listas, heuristic_time = MTZ2(datos)
        
        for lista, t in zip(listas, T_index_prima):         # [[3], [1, 2], [0, 4, 6, 5]]
    
            if len(lista) == 1:
                mugt[lista[0]+1, t].start = 1
                ugt[lista[0]+1, t].start = 1
                vgt[lista[0]+1, t].start = 1
            
                for g1 in T_index_prima:
                    for g2 in T_index_prima:
                        if g1 != g2:
                            yggt[g1, g2, t].start = 0
                
                
            
            if len(lista) > 1:
                
                ugt[lista[0]+1, t].start = 1
                mugt[lista[0]+1, t].start = 1
                
                for i in range(1, len(lista)-1):
                    mugt[lista[i]+1, t].start = 1
                    yggt[lista[i]+1, lista[i+1]+1, t].start = 1
                
                vgt[lista[-1]+1, t].start = 1
                mugt[lista[-1]+1, t].start = 1
                
                # mugt[lista[0]+1, t].start = 1
                # ugt[lista[0]+1, t].start = 1
                #
                # for i in range(1, len(lista)-1):
                #     mugt[lista[i]+1, t].start = 1
                #     yggt[lista[i]+1, lista[i+1]+1, t].start = 1
                #
                # mugt[lista[-1]+1, t].start = 1
                # vgt[lista[-1]+1, t].start = 1 
        

    if datos.init == 3:
        
        listas, heuristic_time = MTZ3(datos)
        
        for lista, t in zip(listas, T_index_prima):         # [[3], [1, 2], [0, 4, 6, 5]]
    
            if len(lista) == 1:
                mugt[lista[0], t].start = 1
                ugt[lista[0], t].start = 1
                vgt[lista[0], t].start = 1
            
                for g1 in T_index_prima:
                    for g2 in T_index_prima:
                        if g1 != g2:
                            yggt[g1, g2, t].start = 0
                
                
            
            if len(lista) > 1:
                
                ugt[lista[0], t].start = 1
                mugt[lista[0], t].start = 1
                
                for i in range(1, len(lista)-1):
                    mugt[lista[i], t].start = 1
                    yggt[lista[i], lista[i+1], t].start = 1
                
                vgt[lista[-1], t].start = 1
                mugt[lista[-1], t].start = 1
                
                # mugt[lista[0]+1, t].start = 1
                # ugt[lista[0]+1, t].start = 1
                #
                # for i in range(1, len(lista)-1):
                #     mugt[lista[i]+1, t].start = 1
                #     yggt[lista[i]+1, lista[i+1]+1, t].start = 1
                #
                # mugt[lista[-1]+1, t].start = 1
                # vgt[lista[-1]+1, t].start = 1 
                
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
    
    for t in T_index_prima:
        restricciones = MODEL.addConstrs(gp.quicksum(yggt[g1, g2, t] for g1, g2 in permutations(set, 2)) <= len(set) - 1 for set in list(powerset(T_index_prima)))
        restricciones.Lazy = 2
    
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
    
    

    
    # MODEL.addConstrs(gp.quicksum(ugt[g, t1] for g in T_index_prima for t1 in T_index_prima if t1 >= t)  >= 1 - beta[t] for t in T_index_prima)
    # MODEL.addConstrs(gp.quicksum(vgt[g, t1] for g in T_index_prima for t1 in T_index_prima if t1 >= t)  >= 1 - beta[t] for t in T_index_prima)
    
    ## PREPROCESADO
    prepro = False
    if prepro:
        numero = 3
        conjunto = preprocessing3(datos, numero)
        print(conjunto)
    
        for lista in conjunto:
            for t in T_index_prima:
                MODEL.addConstr(gp.quicksum(mugt[g, t] for g in lista) <= numero-1)
    
        
    
    
    # for g in T_index_prima:
    #     for t in T_index_prima:
    #         MODEL.addConstr(sgt[g, t] <= nG - 1)
    #         MODEL.addConstr(sgt[g, t] >= 0)
    #
    # # Eliminación de subtours
    # MODEL.addConstrs(nG- 1 >= (sgt[g1, t] - sgt[g2, t]) + nG * yggt[g1, g2, t] for g1, g2, t in yggt.keys())
    
    MODEL.addConstrs(difggt[g1, g2, t, dim] >=  Lp[g1, dim] - Rp[g2, dim] for g1, g2, t, dim in difggt.keys())
    MODEL.addConstrs(difggt[g1, g2, t, dim] >= -Lp[g1, dim] + Rp[g2, dim] for g1, g2, t, dim in difggt.keys())
    MODEL.addConstrs(gp.quicksum(difggt[g1, g2, t, dim]*difggt[g1, g2, t, dim] for dim in N) <= dggt[g1, g2, t] * dggt[g1, g2, t] for g1, g2, t in dggt.keys())
    
    MODEL.addConstrs(dggt[g1, g2, t] == eM.estima_BigM_local(data[g1-1], data[g2-1]) for g1, g2, t in dggt.keys() if type(data[g1-1]) is Punto and type(data[g2-1]) is Punto)
    
    # BigM = 10000
    # SmallM = 0
    
    # MODEL.addConstr(beta[2] == 0)
    
    for g1, g2, t in yggt.keys():
    
        comp1 = data[g1-1]
        comp2 = data[g2-1]
    
        BigM = eM.estima_BigM_local(comp1, comp2)
        SmallM = eM.estima_SmallM_local(comp1, comp2)
        
        # SmallM = 0
        # BigM = 100000
        
        # MODEL.addConstr(pggt[g1, g2, t] <= BigM * yggt[g1, g2, t])
        # MODEL.addConstr(pggt[g1, g2, t] <= dggt[g1, g2, t])
        MODEL.addConstr(pggt[g1, g2, t] >= SmallM * yggt[g1, g2, t])
        MODEL.addConstr(pggt[g1, g2, t] >= dggt[g1, g2, t] - BigM * (1 - yggt[g1, g2, t]))
    
    
    # MODEL.addConstrs(ugt[g, t] <= yggt.sum(g, '*') for g, t in ugt.keys())
    # MODEL.addConstrs(vgt[g, t] <= yggt.sum('*', g) for g, t in vgt.keys())
    
    
    # MODEL.addConstrs(1 - ugt.sum(g, '*') == yggt.sum(g, '*') for g in T_index_prima)
    # MODEL.addConstrs(1 - vgt.sum(g, '*') == yggt.sum('*', g) for g in T_index_prima)
    
    
    # MODEL.addConstrs(ugt.sum('*', t) <= 1 for t in T_index_prima)
    # MODEL.addConstrs(vgt.sum('*', t) <= 1 for t in T_index_prima)
    #
    # MODEL.addConstrs(ugt[g, t] + yggt.sum('*', g, t) == vgt[g, t] + yggt.sum(g, '*', t) for g, t in ugt.keys())
    #
    # MODEL.addConstrs(ugt.sum('*', t) == vgt.sum('*', t) for t in T_index_prima)
    
    
    MODEL.addConstrs((difgLt[g, t, dim] >=  xLt[t, dim] - Rp[g, dim]) for g, t, dim in difgLt.keys())
    MODEL.addConstrs((difgLt[g, t, dim] >= -xLt[t, dim] + Rp[g, dim]) for g, t, dim in difgLt.keys())
    
    MODEL.addConstrs(difgLt[g, t, 0]*difgLt[g, t, 0] + difgLt[g, t, 1]*difgLt[g, t, 1] <= dgLt[g, t]*dgLt[g, t] for g, t in dgLt.keys())
    
    # SmallM = 1
    SmallM = 0
    
    BigM = max([np.linalg.norm(np.array(p) - np.array(q)) for comp1 in data for p in comp1.V for comp2 in data for q in comp2.V])
    # BigM = 10000
    
    # MODEL.addConstrs(pgLt[g, t] <= BigM * ugt[g, t] for g, t in pgLt.keys())
    # MODEL.addConstrs(pgLt[g, t] <= dgLt[g, t] for g, t in pgLt.keys())
    for g, t in pgLt.keys():
        comp = data[g-1]
        if type(comp) is Poligonal:
            # BigM = datos.capacity*vC - comp.alpha*comp.longitud
            MODEL.addConstr(pgLt[g, t] >= SmallM * ugt[g, t])
            MODEL.addConstr(pgLt[g, t] >= dgLt[g, t] - BigM * (1 - ugt[g, t]))
    
    # MODEL.addConstr(gp.quicksum(BigM*(1-ugt[g, t]) - dgLt[g, t] for g, t in ugt.keys()) <= gp.quicksum(dgLt[g, t] for g, t in ugt.keys()))
    # MODEL.addConstr(gp.quicksum(BigM*(1-ugt[g, t]) - dgLt[g, t] for g, t in ugt.keys()) <= gp.quicksum(BigM*ugt[g, t] for g, t in ugt.keys()))
    #
    # MODEL.addConstr(gp.quicksum(BigM*(1-vgt[g, t]) - dgRt[g, t] for g, t in ugt.keys()) <= gp.quicksum(dgRt[g, t] for g, t in ugt.keys()))
    # MODEL.addConstr(gp.quicksum(BigM*(1-vgt[g, t]) - dgRt[g, t] for g, t in ugt.keys()) <= gp.quicksum(BigM*vgt[g, t] for g, t in ugt.keys()))
    
    
    # MODEL.addConstrs(BigM*(1 - ugt[g, t]) <= dgLt[g, t] for g, t in pgLt.keys())
    # MODEL.addConstrs(BigM*(1 - vgt[g, t]) <= dgRt[g, t] for g, t in pgLt.keys())
    
    
    MODEL.addConstrs((difgRt[g, t, dim] >=   Lp[g, dim] - xRt[t, dim]) for g, t, dim in difgRt.keys())
    MODEL.addConstrs((difgRt[g, t, dim] >=  -Lp[g, dim] + xRt[t, dim]) for g, t, dim in difgRt.keys())
    
    MODEL.addConstrs(difgRt[g, t, 0]*difgRt[g, t, 0] + difgRt[g, t, 1]*difgRt[g, t, 1] <= dgRt[g, t]*dgRt[g, t] for g, t in dgRt.keys())
    
    # MODEL.addConstrs(pgRt[g, t] <= BigM * vgt[g, t] for g, t in pgRt.keys())
    # MODEL.addConstrs(pgRt[g, t] <= dgRt[g, t] for g, t in pgRt.keys())
    
    for g, t in pgRt.keys():
        comp = data[g-1]
        if type(comp) is Poligonal:
            # BigM = datos.capacity*vC - comp.alpha*comp.longitud
            MODEL.addConstr(pgRt[g, t] >= SmallM * vgt[g, t])
            MODEL.addConstr(pgRt[g, t] >= dgRt[g, t] - BigM * (1 - vgt[g, t]))
    
    MODEL.addConstrs((difRLt[t, dim] >=   xRt[t, dim] - xLt[t + 1, dim]) for t in T_index_primaprima for dim in range(2))
    MODEL.addConstrs((difRLt[t, dim] >=  -xRt[t, dim] + xLt[t + 1, dim]) for t in T_index_primaprima for dim in range(2))
    MODEL.addConstrs(difRLt[t, 0]*difRLt[t, 0] + difRLt[t, 1] * difRLt[t, 1] <= dRLt[t] * dRLt[t] for t in dRLt.keys())
    
    MODEL.addConstrs((difLRt[t, dim] >=   xLt[t, dim] - xRt[t, dim]) for t, dim in difLRt.keys())
    MODEL.addConstrs((difLRt[t, dim] >= - xLt[t, dim] + xRt[t, dim]) for t, dim in difLRt.keys())
    MODEL.addConstrs(difLRt[t, 0]*difLRt[t, 0] + difLRt[t, 1] * difLRt[t, 1] <= dLRt[t] * dLRt[t] for t in dLRt.keys())
    
    
    for g, t in mugt.keys():
        comp = data[g-1]
        if type(comp) is Poligonal:
            SmallM = comp.alpha*comp.longitud
            # SmallM = 0
            BigM = comp.longitud
            
            # MODEL.addConstr(pgt[g, t] <= BigM * dt[g])
            # MODEL.addConstr(pgt[g, t] <= dt[g])
    
            MODEL.addConstr(pgt[g, t] >= SmallM * mugt[g, t])
            MODEL.addConstr(pgt[g, t] >= dt[g] - BigM* (1 - mugt[g, t]))
        # if type(comp) is Punto:
        #     SmallM = 0
        #     BigM = 0
        #     MODEL.addConstr(pgt[g, t] >= SmallM * mugt[g, t])
        #     MODEL.addConstr(pgt[g, t] >= dt[g] - BigM* (1 - mugt[g, t]))
    
    MODEL.addConstrs((pgLt.sum('*', t) + pgt.sum('*', t) + gp.quicksum(pggt[g1, g2, t] for g1 in T_index_prima for g2 in T_index_prima if g1 != g2) + pgRt.sum('*', t))/vD <= dLRt[t]/vC for t in T_index_prima)
    #
    # MODEL.addConstrs((gp.quicksum(pgLit[g, i, t] for i in grafos[g-1].aristas) + pgij.sum(g, '*', '*') +  gp.quicksum(pgi[g, i]*grafos[g-1].longaristas[i // 100 - 1, i % 100] for i in grafos[g-1].aristas) + gp.quicksum(pgRit[g, i, t] for i in grafos[g-1].aristas))/vD <= dLRt[t]/vC + BigM*(1- gp.quicksum(ugit[g, i, t] for i in grafos[g-1].aristas)) for t in T_index_prima for g in T_index_prima)

    # MODEL.addConstrs((pgLt[g, t] + pgt[g, t] + pggt.sum('*', '*', t) + pgRt[g, t])/vD <= dLRt[t]/vC + BigM*(1 - mugt[g, t]) for g in T_index_prima for t in T_index_prima)
    
    MODEL.addConstrs((dLRt[t]/vC <= datos.capacity) for t in dLRt.keys())
    
    for p in T_index_prima:
        comp = data[p-1]
        print(data)
        if type(comp) is Poligonal:
            # MODEL.addConstr(rho[p] - landa[p] == landa_max[p] - landa_min[p], name='u0')
            # si u = 0, entonces landa0 >= landa1
            # MODEL.addConstr(landa_max[p] + landa_min[p] == comp.alpha * comp.num_segmentos, name='u1')
            MODEL.addConstr(landa_max[p] <= comp.num_segmentos * (1 - u[p]), name='u2')
            MODEL.addConstr(landa_min[p] <= comp.num_segmentos * u[p], name='u3')
            MODEL.addConstr(dt[p] >= comp.alpha*comp.longitud)
            
            # print(zsegm1segm2.keys())
            for segm1 in range(comp.num_segmentos):
                for segm2 in range(comp.num_segmentos):
                    MODEL.addConstr(zsegm1segm2[p, segm1, segm2] <= muR[p, segm1])
                    MODEL.addConstr(zsegm1segm2[p, segm1, segm2] <= muL[p, segm2])
                    MODEL.addConstr(zsegm1segm2[p, segm1, segm2] >= muR[p, segm1] + muL[p, segm2] - 1)
                    

                    if segm1 == segm2:
                        MODEL.addConstr(dsegm1segm2[p, segm1, segm2] == (gammaR[p, segm1]-gammaL[p, segm1])*comp.longitudes[segm1])
                    if segm1 < segm2:
                        MODEL.addConstr(dsegm1segm2[p, segm1, segm2] == gammaR[p, segm1]*comp.longitudes[segm1] + sum([comp.longitudes[k] for k in range(segm1+1, segm2)]) + (1-gammaL[p, segm2])*comp.longitudes[segm2])
                    if segm1 > segm2:
                        MODEL.addConstr(dsegm1segm2[p, segm1, segm2] == (1-gammaR[p, segm1])*comp.longitudes[segm1] + sum([comp.longitudes[k] for k in range(segm2+1, segm1)]) + gammaL[p, segm2]*comp.longitudes[segm2])
        
            
                    MODEL.addConstr(psegm1segm2[p, segm1, segm2] <= comp.longitud* zsegm1segm2[p, segm1, segm2])
                    MODEL.addConstr(psegm1segm2[p, segm1, segm2] <= dsegm1segm2[p, segm1, segm2])
                    MODEL.addConstr(psegm1segm2[p, segm1, segm2] >= dsegm1segm2[p, segm1, segm2] - comp.longitud* (1- zsegm1segm2[p, segm1, segm2]))
            MODEL.addConstr(dt[p] == gp.quicksum(psegm1segm2[p, segm1, segm2] for segm1 in range(comp.num_segmentos) for segm2 in range(comp.num_segmentos)))

                    # MODEL.addConstr(pijRLgg[i, j, g1, g2] <= dijRLgg[i, j, g1, g2])
                    # MODEL.addConstr(pijRLgg[i, j, g1, g2] >= dijRLgg[i, j, g1, g2] - ciclo.longitud*(1 - zijRLgg[i, j, g1, g2]))
            # MODEL.addConstr(dt[p] == (landa_max[p] + landa_min[p])*comp.longitud/comp.num_segmentos)
            # MODEL.addConstr(dt[p] == comp.longitud * comp.alpha)
            
            # print(comp.alpha*comp.longitud)
            # MODEL.addConstrs(rho[p] - punto >= gammaR[p, punto] - comp.num_puntos*(1 - muR[p, punto-1]) for punto in range(1, comp.num_puntos))
            # MODEL.addConstrs(rho[p] - punto <= gammaR[p, punto] + comp.num_puntos*(1 - muR[p, punto-1]) for punto in range(1, comp.num_puntos))
            MODEL.addConstr(gammaR[p, 0] <= muR[p, 0])
            MODEL.addConstrs(gammaR[p, punto] <= muR[p, punto-1] + muR[p, punto] for punto in range(1, comp.num_segmentos))
            MODEL.addConstr(gammaR[p, comp.num_segmentos] <= muR[p, comp.num_segmentos-1])
            MODEL.addConstr(muR.sum(p, '*') == 1)
            MODEL.addConstr(gammaR.sum(p, '*') == 1)
            MODEL.addConstrs(Rp[p, dim] == gp.quicksum(gammaR[p, punto]*comp.V[punto][dim] for punto in range(comp.num_puntos)) for dim in N)
    
            # MODEL.addConstrs(landa[p] - punto >= gammaL[p, punto] - comp.num_puntos*(1 - muL[p, punto-1]) for punto in range(1, comp.num_puntos))
            # MODEL.addConstrs(landa[p] - punto <= gammaL[p, punto] + comp.num_puntos*(1 - muL[p, punto-1]) for punto in range(1, comp.num_puntos))
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
            
    
            # MODEL.addConstr(rho[p] == gp.quicksum(j*muR[p, j] + gammaR[p, j] for j in range(comp.num_segmentos)))
            # # MODEL.addConstr(gammaR[p, 0] <= muR[p, 0])
            # # MODEL.addConstrs(gammaR[p, j] <= muR[p, j-1] + muR[p, j] for j in range(1, comp.num_segmentos))
            # # MODEL.addConstr(gammaR[p, comp.num_segmentos-1] <= muR[p, comp.num_segmentos-1])
            # MODEL.addConstrs(gammaR[p, j] <= muR[p, j] for j in range(comp.num_segmentos))
            # MODEL.addConstrs(pR[p, j] >= muR[p, j] + gammaR[p, j] - 1 for j in range(comp.num_segmentos))
            # MODEL.addConstrs(pR[p, j] <= muR[p, j] for j in range(comp.num_segmentos))
            # MODEL.addConstrs(pR[p, j] <= gammaR[p, j] for j in range(comp.num_segmentos))
            # MODEL.addConstr(muR.sum(p, '*') == 1)
            # # MODEL.addConstr(gammaR.sum(p, '*') == 1)
            # MODEL.addConstrs(Rp[p, dim] == gp.quicksum(muR[p, j]*comp.V[j][dim] + pR[p, j] * (comp.V[j+1][dim] - comp.V[j][dim]) for j in range( comp.num_segmentos)) for dim in N)
            #
            # MODEL.addConstr(landa[p] == gp.quicksum(j*muL[p, j] + gammaL[p, j] for j in range(comp.num_segmentos)))
            # # MODEL.addConstr(gammaL[p, 0] <= muL[p, 0])
            # # MODEL.addConstrs(gammaL[p, j] <= muL[p, j-1] + muL[p, j] for j in range(1, comp.num_segmentos))
            # # MODEL.addConstr(gammaL[p, comp.num_segmentos-1] <= muL[p, comp.num_segmentos-1])
            # MODEL.addConstrs(gammaL[p, j] <= muL[p, j] for j in range(comp.num_segmentos))
            # MODEL.addConstrs(pL[p, j] >= muL[p, j] + gammaL[p, j] - 1 for j in range(comp.num_segmentos))
            # MODEL.addConstrs(pL[p, j] <= muL[p, j] for j in range(comp.num_segmentos))
            # MODEL.addConstrs(pL[p, j] <= gammaL[p, j] for j in range(comp.num_segmentos))
            # MODEL.addConstr(muL.sum(p, '*') == 1)
            # # MODEL.addConstr(gammaL.sum(p, '*') == 1)
            # MODEL.addConstrs(Lp[p, dim] == gp.quicksum(muL[p, j]*comp.V[j][dim] + pL[p, j] * (comp.V[j+1][dim] - comp.V[j][dim]) for j in range( comp.num_segmentos)) for dim in N)
    
    # MODEL.addConstrs((pgRt.sum('*', t) + gp.quicksum(np.linalg.norm(P[g1-1].V - P[g2-1].V)*yggt[g1, g2, t] for g1 in T_index_prima for g2 in T_index_prima if g1 != g2) + pgLt.sum('*', t))/vD <= 30 for t in T_index_prima)
    
    MODEL.addConstrs(xLt[0, dim] == datos.orig[dim] for dim in N)
    MODEL.addConstrs(xRt[0, dim] == datos.orig[dim] for dim in N)
    #
    MODEL.addConstrs(xLt[datos.m+1, dim] == datos.dest[dim] for dim in N)
    MODEL.addConstrs(xRt[datos.m+1, dim] == datos.dest[dim] for dim in N)
    
    # MODEL.addConstr(gp.quicksum(3*dLRt[index] for index in dLRt.keys()) + gp.quicksum(3*dRLt[index] for index in dRLt.keys()) + gp.quicksum(pgRt[index] for index in pgRt.keys()) + gp.quicksum(pgLt[index] for index in pgLt.keys()) + gp.quicksum(pggt[g1, g2, t] for g1, g2, t in yggt.keys()) + gp.quicksum(pgt[index] for index in pgt.keys()) >= 100)
    # MODEL.addConstrs(v.sum('*', e, '*') == 1 for e in E_index) 944 161 
    # MODEL.addConstrs(z.sum(e, '*', '*') == 1 for e in E_index)
    
    MODEL.update()
    
    # Funcion objetivo
    # + gp.quicksum(0.5*pgLt[index] for index in pgLt.keys()) + gp.quicksum(0.5*pgRt[index] for index in pgRt.keys())
    # objective = gp.quicksum(dRLt[index] for index in dRLt.keys()) + gp.quicksum(dLRt[index] for index in dLRt.keys()) + gp.quicksum(pgLt[index] for index in dgLt.keys()) + gp.quicksum(pgRt[index] for index in dgRt.keys())
    
    objective = gp.quicksum(3*dLRt[index] for index in dLRt.keys()) + gp.quicksum(3*dRLt[index] for index in dRLt.keys()) + gp.quicksum(pgRt[index] for index in pgRt.keys()) + gp.quicksum(pgLt[index] for index in pgLt.keys()) + gp.quicksum(pggt[g1, g2, t] for g1, g2, t in yggt.keys()) + gp.quicksum(pgt[index] for index in pgt.keys())
    # objective = gp.quicksum(3*dLRt[index]/datos.vC for index in dLRt.keys()) + gp.quicksum(3*dRLt[index]/datos.vC for index in dRLt.keys()) + gp.quicksum(pgRt[index]/datos.vD for index in pgRt.keys()) + gp.quicksum(pgLt[index]/datos.vD for index in pgLt.keys()) + gp.quicksum(pggt[g1, g2, t]/datos.vD for g1, g2, t in yggt.keys()) + gp.quicksum(pgt[index]/datos.vD for index in pgt.keys())
    
    MODEL.setObjective(objective, GRB.MINIMIZE)
    
    MODEL.update()
    
    # MODEL.setParam('TimeLimit', datos.tmax)
    
    MODEL.Params.Threads = 5
    MODEL.Params.TimeLimit = datos.tmax
    MODEL.Params.LazyConstraints = 1
    # MODEL.Params.MIPFocus = 3

    MODEL.Params.LogFile = 'solution'+str(log)+'.log'
    # MODEL.Params.FeasibilityTol = 1e-3
    MODEL._yggt = yggt
    MODEL._mugt = mugt
    MODEL._ugt = ugt
    MODEL._vgt = vgt
    
    MODEL.update()
    # MODEL.Params.SolutionLimit = 3
    
    # MODEL = MODEL.relax()
    
    # Optimizamos
    # datos.init = 0
    if datos.init:
        MODEL.optimize(callback)
    else:
        MODEL.optimize()


    # MODEL.optimize()
    # MODEL.optimize(subtourelim)
    


    
    if MODEL.Status == 3:
        result = []
        result =  [np.nan, np.nan, np.nan, np.nan, np.nan]
        
        MODEL.computeIIS()
        MODEL.write('infactible.ilp')
        
        return result

    MODEL.write('aver.lp')
    MODEL.write('solution.sol')
       
    if MODEL.SolCount == 0:
        result = []
        result =  [np.nan, np.nan, np.nan, np.nan, np.nan]
        
        if datos.init:
            result.append(np.nan)
            result.append(np.nan)
            
        
        # MODEL.computeIIS()
        # MODEL.write('sinsol.ilp')
        
        return result
    
    MODEL.write('./sol_files/multitarget.sol')
    
    vals_ugt = MODEL.getAttr('x', ugt)
    
    selected_ugt = gp.tuplelist(e for e in vals_ugt if vals_ugt[e] > 0.5)
    
    print('ugt')
    print(selected_ugt)
    
    vals_vgt = MODEL.getAttr('x', vgt)
    
    selected_vgt = gp.tuplelist(e for e in vals_vgt if vals_vgt[e] > 0.5)
    
    print('vgt')
    print(selected_vgt)
    
    vals_yggt = MODEL.getAttr('x', yggt)
    print(vals_yggt)
    
    selected_yggt = gp.tuplelist(e for e in vals_yggt if vals_yggt[e] > 0.5)
    
    print('yggt')
    print(selected_yggt)
    
    vals_mugt = MODEL.getAttr('x', mugt)
    
    selected_mugt = gp.tuplelist(e for e in vals_mugt if vals_mugt[e] > 0.5)
    
    print('mugt')
    print(selected_mugt)
    
    vals_beta = MODEL.getAttr('x', beta)
    
    selected_beta = gp.tuplelist(e for e in vals_beta if vals_beta[e] > 0.5)
    
    print('beta')
    print(selected_beta)
    
    
    path = []
    # path.append(0)
    
    for t in T_index_prima:
        tripleta = selected_ugt.select('*', t)
        if tripleta:
            path.append(tripleta[0][1])
    
    # path.append(nG+1)
    # print(path)
    
    if datos.show:
        ind = 0
        path_C = []
        paths_D = []
    
        path_C.append(datos.orig)
        path_C.append([xLt[0, 0].X, xLt[0, 1].X])
        for t in path:
            #    if ind < datos.m:
            path_C.append([xLt[t, 0].X, xLt[t, 1].X])
            path_D = []
            path_D.append([xLt[t, 0].X, xLt[t, 1].X])
            index_g = g
            index_t = t
            for g, ti in selected_ugt:
                if ti == t:
                    index_g = g
                    index_t = ti
        
            # path_D.append([Lgi[index_g, index_i, 0].X, Lgi[index_g, index_i, 1].X])
            limite = sum([1 for g, ti in selected_mugt if ti == index_t])
            path_D.append([Rp[index_g, 0].X, Rp[index_g, 1].X])
            path_D.append([Lp[index_g, 0].X, Lp[index_g, 1].X])
            count = 1
            while count < limite:
                for g1, g2, ti in selected_yggt:
                    if index_t == ti and index_g == g1:
                        count += 1
                        index_g = g2
                        print(index_g)
                        # path_D.append([Rgi[index_g, index_i, 0].X, Rgi[index_g, index_i, 1].X])
                        path_D.append([Rp[index_g, 0].X, Rp[index_g, 1].X])
                        path_D.append([Lp[index_g, 0].X, Lp[index_g, 1].X])
        
            path_D.append([xRt[t, 0].X, xRt[t, 1].X])
            paths_D.append(path_D)
            path_C.append([xRt[t, 0].X, xRt[t, 1].X])
        
        path_C.append([xLt[datos.m+1, 0].X, xLt[datos.m+1, 1].X])
        
        fig, ax = plt.subplots()
        
        # ax.axis('equal')
        ax.set_aspect('equal')
        plt.axis([0, 100, 0, 100])
        
        # plt.axis('equal')
        
        #
        # path_C = []
        
        plt.plot(xLt[0, 0].X, xLt[0, 1].X, 's', alpha = 1, markersize=10, color='black')
        plt.plot(xLt[nG+1, 0].X, xLt[nG+1, 1].X, 's', alpha = 1, markersize=10, color='black')
        
        for t in path:
            # path_C.append([xLt[t, 0].X, xLt[t, 1].X])
            # path_C.append([xRt[t, 0].X, xRt[t, 1].X])
            # plt.plot(xLt[t, 0].X, xLt[t, 1].X, 's', alpha = 1, markersize=5, color='black')
            # plt.plot(xLt[1, 0].X, xLt[1, 1].X, 'o', alpha = 1, markersize=5,  color='red')
        
            # if t == 0:
                # ax.annotate("orig", xy = (xLt[t, 0].X-2, xLt[t, 1].X+1), fontsize = 15)
            if t > 0 and t < nG+1:
                plt.plot(xLt[t, 0].X, xLt[t, 1].X, 'o', alpha = 1, markersize = 5,  color='red')
                plt.plot(xRt[t, 0].X, xRt[t, 1].X, 'o', alpha = 1, markersize = 5,  color='red')
                # ax.annotate("$x_R^{t}$".format(t = t), xy = (xRt[t, 0].X+1.5, xRt[t, 1].X), fontsize = 15)
                # ax.annotate("$x_L^{t}$".format(t = t), xy = (xLt[t, 0].X-3, xLt[t, 1].X), fontsize = 15)
            # if t == nG+1:
            #     plt.plot(xLt[t, 0].X, xLt[t, 1].X, 's', alpha = 1, markersize=10, color='black')
                # ax.annotate("dest", xy = (xLt[t, 0].X+0.5, xLt[t, 1].X+1), fontsize = 15)
        
        ax.add_artist(Polygon(path_C, fill=False, animated=False, lw = 2, linestyle='-', alpha=1, color='blue'))
        
        for p in T_index_prima:
            plt.plot(Rp[p, 0].X, Rp[p, 1].X, 'o', markersize=5, color='red')
                # ax.annotate("$R_" + str(g) + "^{" + str((first, second)) + "}$", xy = (Rig[i, g, 0].X+0.75, Rig[i, g, 1].X+0.75))    
            plt.plot(Lp[p, 0].X, Lp[p, 1].X, 'o', markersize=5, color='red')
        
        for pathd in paths_D:
            ax.add_artist(Polygon(pathd, fill=False, closed=False, lw = 2, animated=False, alpha=1, color='red'))
        
        for g in T_index_prima:
            ax.add_artist(data[g-1].artist)
        
        plt.title(str(datos.m) + "coordinate ")
        # string = 'imagenes/' + str(m) + "-XPPN - MTZ - Mode " + \
            # str(modo) + " - Radii - " + str(datos.r) + ".png"
        import tikzplotlib
        
        tikzplotlib.save('Instance{b}-{d}.tex'.format(b = datos.m, d = datos.capacity), encoding = 'utf-8')

        plt.savefig('Instance{b}-{d}.png'.format(b = datos.m, d = datos.capacity))
    
    # plt.show()
    

    result = []

    result.append(MODEL.getAttr('MIPGap'))
    result.append(MODEL.Runtime)
    result.append(MODEL.getAttr('NodeCount'))
    result.append(MODEL.ObjVal)
    # result.append(sec)
    
    infile = r"./solution" +str(log)+".log"

    important = []
    keep_phrases = ['Processed MIP start']
    
    with open(infile) as f:
        f = f.readlines()
    
    for line in f:
        for phrase in keep_phrases:
            if phrase in line:
                important.append(line)
                break
    
    print(important)
    # words = important[0].split()
    
    # heuristic_time = float(words[-2][0:-1])
    
    if datos.init:
        # result.append(heuristic_time)
        result.append(MODEL._startobjval)
        result.append(MODEL._starttime)
        
        print(result)
    # polygon_camion = Polygon(path_camion, fill=False, linestyle=':', alpha=1, color='red')
    # ax.add_patch(polygon_camion)
    
    

    
    return result
    
    #plt.close()
    # plt.show()
# MultiTargetPolygonal(datos, 1)