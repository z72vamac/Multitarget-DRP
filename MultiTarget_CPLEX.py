"""Tenemos un conjunto E de entornos ugt un conjunto de poligonales P de las que queremos recorrer un porcentaje alfa p . Buscamos
   un tour de mínima distancia que alterne poligonal-entorno ugt que visite todas las poligonales"""
   
" Esto permite preprocesado de buscar conjuntos de variables ys que no pueden ser uno simultaneamente"

# Incluimos primero los paquetes
# import cplex
import docplex.mp.model as cpx
import docplex.mp.solution as cpxsol
from docplex.mp.model import Model
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
import time
from cplex.callbacks import MIPInfoCallback

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

np.random.seed(12)

# m = 10

# datos = Data([], m = 10,
#                  r = 4,
#                  capacity = 1000,
#                  modo = 2,
#                  tmax = 100,
#                  init = 1,
#                  show = True,
#                  seed = 2)
# datos.generar_muestra()
#
# data = datos.mostrar_datos()

# n = 2
# datos.orig = [0, 0]
# datos.dest = [0, 0]
                
def MultiTargetPolygonal_CPLEX(datos, log):
    vD = datos.vD
    vC = datos.vC
    
    n = 2
    nG = datos.m

    
    data = datos.mostrar_datos()
    global sec
    global gap
    global timeused
    
    class TimeLimitCallback(MIPInfoCallback):

        def __call__(self):
            if self.has_incumbent():
                # self.gap = 100.0 * self.get_MIP_relative_gap()
                self.objective_value = self.get_incumbent_objective_value()
                self.timeused = self.get_time() - self.starttime
                print("First feasible solution at", self.timeused, "sec., objective_value =",
                      self.objective_value, "%, quitting.")
                # self.abort()
                # self.aborted = True
                # self.abort()
    
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
                            model.cbLazy(MODEL.sum(model._yggt[g1, g2, t] for g1, g2 in permutations(tour, 2)) <= len(tour) - 1)
                            print(MODEL.sum(model._yggt[g1, g2, t] for g1, g2 in permutations(tour, 2)))
                            global sec
                            sec += 1
    # nE = 2
    # np.random.seed(15)
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
    MODEL = Model(name="MultiTarget_Cplex", log_output = True)
    MODEL.context.cplex_parameters.threads = 5
    
    
    # Variable binaria ugt = 1 si se entra por el grafo g en la etapa t
    
    ugt_index = []
    difgt_index = []
    
    for g in T_index_prima:
        for t in T_index_prima:
            ugt_index.append((g, t))
            for dim in range(2):
                difgt_index.append((g, t, dim))
    
    ugt = MODEL.binary_var_dict(ugt_index , name='ugt')
    
    
    # Variable binaria ugt = 1 si se entra por el grafo g en la etapa t
    
    mugt_index = ugt_index
    
    mugt = MODEL.binary_var_dict(mugt_index , name='mugt')
    kt = MODEL.continuous_var_dict(T_index_prima , name='kt')
    

    pgt = MODEL.continuous_var_dict(mugt_index , name = 'pgt')
    
    # Variable continua no negativa dgLt que indica la distancia desde el punto de lanzamiento hasta el grafo g.
    
    dgLt_index = ugt_index
    
    dgLt = MODEL.continuous_var_dict(dgLt_index , lb=0.0, name='dgLt')
    difgLt = MODEL.continuous_var_dict(difgt_index , lb=0.0, name='difgLt')
    
    # Variable continua no negativa pgLit = ugit * dgLit
    pgLt_index = ugt_index
    
    pgLt = MODEL.continuous_var_dict(pgLt_index , lb=0.0, name='pgLt')
    
    # Variable binaria yggt = 1 si voy de g1 a g2.
    yggt_index = []
    
    difggt_index = []
    for g1 in T_index_prima:
        for g2 in T_index_prima:
            if g1 != g2:
                for t in T_index_prima:
                    yggt_index.append((g1, g2, t))
                    for dim in range(2):
                        difggt_index.append((g1, g2, t, dim))
    
    yggt = MODEL.binary_var_dict(yggt_index , name = 'yggt')
    dggt = MODEL.continuous_var_dict(yggt_index , name = 'Dggt')
    difggt = MODEL.continuous_var_dict(difggt_index , name = 'difggt')
    pggt = MODEL.continuous_var_dict(yggt_index , name = 'pggt')
    
    
    # Variable continua sgt que indica el orden en la etapa
    sgt_index = []
    
    for t in T_index_prima:
        for g in T_index_prima:
            sgt_index.append((g, t))
    
    sgt = MODEL.continuous_var_dict(sgt_index, name = 'sgt')
    
    # Variable binaria vgt = 1 si en la etapa t salimos por el grafo g
    vgt_index = ugt_index
    
    vgt = MODEL.binary_var_dict(vgt_index , name='vgt')
    
    # Variable continua no negativa dgRit que indica la distancia desde el punto de salida del segmento sgi hasta el
    # punto de recogida del camion
    dgRt_index = ugt_index
    
    dgRt = MODEL.continuous_var_dict(dgRt_index , lb=0.0, name='dgRt')
    difgRt = MODEL.continuous_var_dict(difgt_index, lb=0.0, name='difgRt')
    
    # Variable continua no negativa pgRt = ugt * dgRt
    pgRt_index = ugt_index
    
    pgRt = MODEL.continuous_var_dict(pgRt_index , lb=0.0, name='pgRt')
    
    # Variable continua no negativa dRLt que indica la distancia entre el punto de recogida en la etapa t ugt el punto de
    # salida para la etapa t+1
    dRLt_index = T_index_primaprima
    
    difRLt_index = []
    for t in T_index_primaprima:
        for dim in range(2):
            difRLt_index.append((t, dim))
            
    dRLt = MODEL.continuous_var_dict(dRLt_index , lb=0.0, name='dRLt')
    difRLt = MODEL.continuous_var_dict(difRLt_index, lb=0.0, name='difRLt')
    
    # Variable continua no negativa dLRt que indica la distancia que recorre el camión en la etapa t mientras el dron se mueve
    dLRt_index = T_index
    
    difLRt_index = []
    for t in T_index:
        for dim in range(2):
            difLRt_index.append((t, dim))
    
    dLRt = MODEL.continuous_var_dict(dLRt_index , lb=0.0, name='dLRt')
    difLRt = MODEL.continuous_var_dict(difLRt_index , lb=0.0, name='difLRt')
    
    # Variables que modelan los puntos de entrada o recogida
    # xLt: punto de salida del dron del camion en la etapa t
    xLt_index = []
    
    for t in T_index:
        for dim in range(2):
            xLt_index.append((t, dim))
    
    xLt = MODEL.continuous_var_dict(xLt_index , name='xLt')
    
    # xRt: punto de recogida del dron del camion en la etapa t
    xRt_index = []
    
    for t in T_index:
        for dim in range(2):
            xRt_index.append((t, dim))
    
    xRt = MODEL.continuous_var_dict(xRt_index , name='xRt')
    
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
    
    
    landa = MODEL.continuous_var_dict(landa_index , name='landa')
    rho = MODEL.continuous_var_dict(rho_index , name='rho')
    
    gammaR = MODEL.continuous_var_dict(gammaR_index , lb=0.0, ub=1.0, name='gammaR')
    gammaL = MODEL.continuous_var_dict(gammaL_index , lb=0.0, ub=1.0, name='gammaL')
    
    muR = MODEL.binary_var_dict(muR_index , name='muR')
    muL = MODEL.binary_var_dict(muL_index , name='muL')
        
    u = MODEL.binary_var_dict(u_index , name='u')
    dt = MODEL.continuous_var_dict(T_index_prima , name = 'dt')
    
    landa_min = MODEL.continuous_var_dict(u_index , lb=0.0, name='landa_min')
    landa_max = MODEL.continuous_var_dict(u_index , lb=0.0, name='landa_max')
    
    Rp = MODEL.continuous_var_dict(Rp_index , name = 'Rp')
    Lp = MODEL.continuous_var_dict(Lp_index , name = 'Lp')
    
    beta = MODEL.binary_var_dict(T_index_prima , name = 'beta')
        
    # mugt_inicial, heur_time = heuristic(datos)

    if datos.init:
    
        tour, heur_time = MTZ(datos)
        
        tour = tour[1:-1]
        
        print(tour)
        warmstart = MODEL.new_solution()
        
        
        
        for t in T_index_prima:
            warmstart.add_var_value(mugt[tour[t-1], t], 1)
            warmstart.add_var_value(ugt[tour[t-1], t], 1)
            warmstart.add_var_value(vgt[tour[t-1], t], 1)
            
        for g1, g2, t in yggt.keys():
            warmstart.add_var_value(yggt[g1, g2, t],  0)
        

        MODEL.add_mip_start(warmstart)
        

        #
        # for g, t in mugt.keys():
        #     if (g, t) in mugt_start:
        #         mugt[g, t].start = 1
        #     else:
        #         mugt[g, t].start = 0
        #

        # MODEL.read('./sol_files/multitarget.sol')
        
        
    # (1), (2): for each operation the drone only can enter and exit, respectively,by one target
    for t in T_index_prima:
        MODEL.add_constraint((MODEL.sum(ugt[g, t] for g in T_index_prima) <= 1), ctname = 'constr1')
        MODEL.add_constraint((MODEL.sum(vgt[g, t] for g in T_index_prima) <= 1), ctname = 'constr2')
    
    # (3): every target will be visited in some operation
    for g in T_index_prima:
        MODEL.add_constraint((MODEL.sum(mugt[g, t] for t in T_index_prima) == 1), ctname = 'constr3')
    
    # (4): if target t is visited by the drone for the operation o, one of two alternative situations must occur: either t is the first target 
    # for the operation o or target t is visited by the drone after visiting another target t' for the operation o. 
    for g in T_index_prima:
        for t in T_index_prima:
            MODEL.add_constraint((mugt[g, t] - ugt[g, t] == MODEL.sum(yggt[g1, g, t] for g1 in T_index_prima if g1 != g)), ctname = 'constr4')
    
    # (5): state that if the target t for the operation o is visited by the drone, either t is the last target of the operation, 
    # or the drone must move to another target t' of the operation o after visiting target t.
    for g in T_index_prima:
        for t in T_index_prima:
            MODEL.add_constraint((mugt[g, t] - vgt[g, t] == MODEL.sum(yggt[g, g1, t] for g1 in T_index_prima if g1 != g)), ctname = 'constr5')
    
    
    # (6): Subtour elimination constraints
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))
    
    for t in T_index_prima:
        MODEL.add_lazy_constraints((MODEL.sum(yggt[g1, g2, t] for g1, g2 in permutations(set, 2)) <= len(set) - 1 for set in list(powerset(T_index_prima))))
    
    # Monotonicity
    for t in T_index_prima[0:-1]:
        MODEL.add_constraint((beta[t] <= beta[t+1]), ctname = 'monotonicity')
        
    # Definition of kt
    for t in T_index_prima:
        kt[t] = MODEL.sum(mugt[g, t] for g in T_index_prima)
  
    
    # (VI-1)
    for t in T_index_prima:
        MODEL.add_constraint((MODEL.sum(kt[t1] for t1 in T_index_prima if t1 < t) >= datos.m*beta[t]), ctname = 'VI-1')
    

    # (VI-2)
    for t in T_index_prima:
        MODEL.add_constraint((kt[t] >= 1 - beta[t]), ctname = 'VI-2')
        
    # MODEL.add_constraint(MODEL.sum(ugt[g, t1] for g in T_index_prima for t1 in T_index_prima if t1 >= t)  >= 1 - beta[t] for t in T_index_prima)
    # MODEL.add_constraint(MODEL.sum(vgt[g, t1] for g in T_index_prima for t1 in T_index_prima if t1 >= t)  >= 1 - beta[t] for t in T_index_prima)
    
    ## PREPROCESADO
    # prepro = False
    # if prepro:
    #     numero = 2
    #     conjunto = preprocessing3(datos, numero)
    #     print(conjunto)
    #
    #     for lista in conjunto:
    #         for t in T_index_prima:
    #             MODEL.add_constraint(MODEL.sum(mugt[g, t] for g in lista) <= numero-1)
    
        
    
    
    # for g in T_index_prima:
    #     for t in T_index_prima:
    #         MODEL.add_constraint(sgt[g, t] <= nG - 1)
    #         MODEL.add_constraint(sgt[g, t] >= 0)
    #
    # # Eliminación de subtours
    # MODEL.add_constraint(nG- 1 >= (sgt[g1, t] - sgt[g2, t]) + nG * yggt[g1, g2, t] for g1, g2, t in yggt.keys())
    
    MODEL.add_constraints(difggt[g1, g2, t, dim] >=  Lp[g1, dim] - Rp[g2, dim] for g1, g2, t, dim in difggt.keys())
    MODEL.add_constraints(difggt[g1, g2, t, dim] >= -Lp[g1, dim] + Rp[g2, dim] for g1, g2, t, dim in difggt.keys())
    
    for g1, g2, t in yggt.keys():
        MODEL.add_constraint(MODEL.sum(difggt[g1, g2, t, dim]*difggt[g1, g2, t, dim] for dim in range(2)) <= dggt[g1, g2, t] * dggt[g1, g2, t])
    
    # BigM = 10000
    # SmallM = 0
    
    # MODEL.add_constraint(beta[2] == 0)
    
    for g1, g2, t in yggt.keys():
    
        comp1 = data[g1-1]
        comp2 = data[g2-1]
    
        BigM = eM.estima_BigM_local(comp1, comp2)
        SmallM = eM.estima_SmallM_local(comp1, comp2)
    
        MODEL.add_constraint(pggt[g1, g2, t] >= SmallM * yggt[g1, g2, t])
        MODEL.add_constraint(pggt[g1, g2, t] >= dggt[g1, g2, t] - BigM * (1 - yggt[g1, g2, t]))
    
    
    # MODEL.add_constraint(ugt[g, t] <= yggt.sum(g, '*') for g, t in ugt.keys())
    # MODEL.add_constraint(vgt[g, t] <= yggt.sum('*', g) for g, t in vgt.keys())
    
    
    # MODEL.add_constraint(1 - ugt.sum(g, '*') == yggt.sum(g, '*') for g in T_index_prima)
    # MODEL.add_constraint(1 - vgt.sum(g, '*') == yggt.sum('*', g) for g in T_index_prima)
    
    
    # MODEL.add_constraint(ugt.sum('*', t) <= 1 for t in T_index_prima)
    # MODEL.add_constraint(vgt.sum('*', t) <= 1 for t in T_index_prima)
    #
    # MODEL.add_constraint(ugt[g, t] + yggt.sum('*', g, t) == vgt[g, t] + yggt.sum(g, '*', t) for g, t in ugt.keys())
    #
    # MODEL.add_constraint(ugt.sum('*', t) == vgt.sum('*', t) for t in T_index_prima)
    
    
    MODEL.add_constraints((difgLt[g, t, dim] >=  xLt[t, dim] - Rp[g, dim]) for g, t, dim in difgLt.keys())
    MODEL.add_constraints((difgLt[g, t, dim] >= -xLt[t, dim] + Rp[g, dim]) for g, t, dim in difgLt.keys())
    
    for g, t in dgLt.keys():
        MODEL.add_constraint(difgLt[g, t, 0]*difgLt[g, t, 0] + difgLt[g, t, 1]*difgLt[g, t, 1] <= dgLt[g, t]*dgLt[g, t])
    
    SmallM = 0
    # BigM = max([np.linalg.norm(np.array(comp[p].V) - np.array(P[p].V)) for p in range(datos.m) for q in range(datos.m)])

    BigM2 = max([np.linalg.norm(np.array(p) - np.array(q)) for comp1 in data for p in comp1.V for comp2 in data for q in comp2.V])
    
    for g, t in pgLt.keys():
        comp = data[g-1]
        if type(comp) is Poligonal:
            BigM = datos.capacity*datos.vD - comp.alpha*comp.longitud
        if type(comp) is Punto:
            BigM = datos.capacity*datos.vD
            
        BigM = BigM2
        MODEL.add_constraint(pgLt[g, t] >= SmallM * ugt[g, t])
        MODEL.add_constraint(pgLt[g, t] >= dgLt[g, t] - BigM * (1 - ugt[g, t]))
    
    # MODEL.add_constraint(MODEL.sum(BigM*(1-ugt[g, t]) - dgLt[g, t] for g, t in ugt.keys()) <= MODEL.sum(dgLt[g, t] for g, t in ugt.keys()))
    # MODEL.add_constraint(MODEL.sum(BigM*(1-ugt[g, t]) - dgLt[g, t] for g, t in ugt.keys()) <= MODEL.sum(BigM*ugt[g, t] for g, t in ugt.keys()))
    #
    # MODEL.add_constraint(MODEL.sum(BigM*(1-vgt[g, t]) - dgRt[g, t] for g, t in ugt.keys()) <= MODEL.sum(dgRt[g, t] for g, t in ugt.keys()))
    # MODEL.add_constraint(MODEL.sum(BigM*(1-vgt[g, t]) - dgRt[g, t] for g, t in ugt.keys()) <= MODEL.sum(BigM*vgt[g, t] for g, t in ugt.keys()))
    
    
    # MODEL.add_constraint(BigM*(1 - ugt[g, t]) <= dgLt[g, t] for g, t in pgLt.keys())
    # MODEL.add_constraint(BigM*(1 - vgt[g, t]) <= dgRt[g, t] for g, t in pgLt.keys())
    
    
    MODEL.add_constraints((difgRt[g, t, dim] >=   Lp[g, dim] - xRt[t, dim]) for g, t, dim in difgRt.keys())
    MODEL.add_constraints((difgRt[g, t, dim] >=  -Lp[g, dim] + xRt[t, dim]) for g, t, dim in difgRt.keys())
    
    for g, t in dgRt.keys():
        MODEL.add_constraint(difgRt[g, t, 0]*difgRt[g, t, 0] + difgRt[g, t, 1]*difgRt[g, t, 1] <= dgRt[g, t]*dgRt[g, t])
    
    for g, t in pgRt.keys():
        comp = data[g-1]
        if type(comp) is Poligonal:
            BigM = datos.capacity*datos.vD - comp.alpha*comp.longitud
        if type(comp) is Punto:
            BigM = datos.capacity*datos.vD
        
        BigM = BigM2
        MODEL.add_constraint(pgRt[g, t] >= SmallM * vgt[g, t])
        MODEL.add_constraint(pgRt[g, t] >= dgRt[g, t] - BigM * (1 - vgt[g, t]))
    
    MODEL.add_constraints((difRLt[t, dim] >=   xRt[t, dim] - xLt[t + 1, dim]) for t in T_index_primaprima for dim in range(2))
    MODEL.add_constraints((difRLt[t, dim] >=  -xRt[t, dim] + xLt[t + 1, dim]) for t in T_index_primaprima for dim in range(2))
    
    for t in dRLt.keys():
        MODEL.add_constraint(difRLt[t, 0]*difRLt[t, 0] + difRLt[t, 1] * difRLt[t, 1] <= dRLt[t] * dRLt[t])
    
    MODEL.add_constraints((difLRt[t, dim] >=   xLt[t, dim] - xRt[t, dim]) for t, dim in difLRt.keys())
    MODEL.add_constraints((difLRt[t, dim] >= - xLt[t, dim] + xRt[t, dim]) for t, dim in difLRt.keys())
    
    for t in dLRt.keys():
        MODEL.add_constraint(difLRt[t, 0]*difLRt[t, 0] + difLRt[t, 1] * difLRt[t, 1] <= dLRt[t] * dLRt[t])
    
    for g, t in mugt.keys():
        comp = data[g-1]
        if type(comp) is Poligonal:
            SmallM = comp.alpha*comp.longitud
            BigM = comp.longitud
        if type(comp) is Punto:
            SmallM = 0
            BigM = 0
        MODEL.add_constraint(pgt[g, t] >= SmallM * mugt[g, t])
        MODEL.add_constraint(pgt[g, t] >= dt[g] - BigM * (1 - mugt[g, t]))
            
    # MODEL.add_constraints(pgt[g, t] >= SmallM * mugt[g, t] for g, t in mugt.keys())
    # MODEL.add_constraints(pgt[g, t] >= dt[g] - BigM * (1 - mugt[g, t]) for g, t in mugt.keys())
    
    
    MODEL.add_constraints((MODEL.sum(pgLt[g, t] for g in T_index_prima) + MODEL.sum(pgt[g, t] for g in T_index_prima) + MODEL.sum(pggt[g1, g2, t] for g1 in T_index_prima for g2 in T_index_prima if g1 != g2) + MODEL.sum(pgRt[g, t] for g in T_index_prima))/vD <= dLRt[t]/vC for t in T_index_prima)
    #
    # MODEL.add_constraint((MODEL.sum(pgLit[g, i, t] for i in grafos[g-1].aristas) + pgij.sum(g, '*', '*') +  MODEL.sum(pgi[g, i]*grafos[g-1].longaristas[i // 100 - 1, i % 100] for i in grafos[g-1].aristas) + MODEL.sum(pgRit[g, i, t] for i in grafos[g-1].aristas))/vD <= dLRt[t]/vC + BigM*(1- MODEL.sum(ugit[g, i, t] for i in grafos[g-1].aristas)) for t in T_index_prima for g in T_index_prima)

    # MODEL.add_constraint((pgLt[g, t] + pgt[g, t] + pggt.sum('*', '*', t) + pgRt[g, t])/vD <= dLRt[t]/vC + BigM*(1 - mugt[g, t]) for g in T_index_prima for t in T_index_prima)
    
    MODEL.add_constraints((dLRt[t]/vC <= datos.capacity) for t in dLRt.keys())
    
    for p in T_index_prima:
        comp = data[p-1]
        if type(comp) is Poligonal:
            MODEL.add_constraint(rho[p] - landa[p] == landa_max[p] - landa_min[p], ctname='u0')
            # si u = 0, entonces landa0 >= landa1
            MODEL.add_constraint(landa_max[p] + landa_min[p] >= comp.alpha * comp.num_segmentos, ctname='u1')
            MODEL.add_constraint(landa_max[p] <= comp.num_segmentos * (1 - u[p]), ctname='u2')
            MODEL.add_constraint(landa_min[p] <= comp.num_segmentos * u[p], ctname='u3')
            MODEL.add_constraint(dt[p] == (landa_max[p] + landa_min[p])*comp.longitud/comp.num_segmentos)
    
            print(comp.alpha*comp.longitud)
            MODEL.add_constraints(rho[p] - punto >= gammaR[p, punto] - comp.num_puntos*(1 - muR[p, punto-1]) for punto in range(1, comp.num_puntos))
            MODEL.add_constraints(rho[p] - punto <= gammaR[p, punto] + comp.num_puntos*(1 - muR[p, punto-1]) for punto in range(1, comp.num_puntos))
            MODEL.add_constraint(gammaR[p, 0] <= muR[p, 0])
            MODEL.add_constraints(gammaR[p, punto] <= muR[p, punto-1] + muR[p, punto] for punto in range(1, comp.num_segmentos))
            MODEL.add_constraint(gammaR[p, comp.num_segmentos] <= muR[p, comp.num_segmentos-1])
            MODEL.add_constraint(MODEL.sum(muR[p, segm] for segm in range(comp.num_segmentos)) == 1)
            MODEL.add_constraint(MODEL.sum(gammaR[p, punto] for punto in range(comp.num_puntos)) == 1)
            MODEL.add_constraints(Rp[p, dim] == MODEL.sum(gammaR[p, punto]*comp.V[punto][dim] for punto in range(comp.num_puntos)) for dim in N)
    
            MODEL.add_constraints(landa[p] - punto >= gammaL[p, punto] - comp.num_puntos*(1 - muL[p, punto-1]) for punto in range(1, comp.num_puntos))
            MODEL.add_constraints(landa[p] - punto <= gammaL[p, punto] + comp.num_puntos*(1 - muL[p, punto-1]) for punto in range(1, comp.num_puntos))
            MODEL.add_constraint(gammaL[p, 0] <= muL[p, 0])
            MODEL.add_constraints(gammaL[p, punto] <= muL[p, punto-1] + muL[p, punto] for punto in range(1, comp.num_segmentos))
            MODEL.add_constraint(gammaL[p, comp.num_segmentos] <= muL[p, comp.num_segmentos-1])
            MODEL.add_constraint(MODEL.sum(muL[p, segm] for segm in range(comp.num_segmentos)) == 1)
            MODEL.add_constraint(MODEL.sum(gammaL[p, punto] for punto in range(comp.num_puntos)) == 1)
            MODEL.add_constraints(Lp[p, dim] == MODEL.sum(gammaL[p, punto]*comp.V[punto][dim] for punto in range(comp.num_puntos)) for dim in N)
        if type(comp) is Punto:
            for dim in N:
                MODEL.add_constraint(Lp[p, dim] == comp.V[dim])
                MODEL.add_constraint(Rp[p, dim] == comp.V[dim])
                MODEL.add_constraint(dt[p] == 0)
            
    
            # MODEL.add_constraint(rho[p] == MODEL.sum(j*muR[p, j] + gammaR[p, j] for j in range(comp.num_segmentos)))
            # # MODEL.add_constraint(gammaR[p, 0] <= muR[p, 0])
            # # MODEL.add_constraint(gammaR[p, j] <= muR[p, j-1] + muR[p, j] for j in range(1, comp.num_segmentos))
            # # MODEL.add_constraint(gammaR[p, comp.num_segmentos-1] <= muR[p, comp.num_segmentos-1])
            # MODEL.add_constraint(gammaR[p, j] <= muR[p, j] for j in range(comp.num_segmentos))
            # MODEL.add_constraint(pR[p, j] >= muR[p, j] + gammaR[p, j] - 1 for j in range(comp.num_segmentos))
            # MODEL.add_constraint(pR[p, j] <= muR[p, j] for j in range(comp.num_segmentos))
            # MODEL.add_constraint(pR[p, j] <= gammaR[p, j] for j in range(comp.num_segmentos))
            # MODEL.add_constraint(muR.sum(p, '*') == 1)
            # # MODEL.add_constraint(gammaR.sum(p, '*') == 1)
            # MODEL.add_constraint(Rp[p, dim] == MODEL.sum(muR[p, j]*comp.V[j][dim] + pR[p, j] * (comp.V[j+1][dim] - comp.V[j][dim]) for j in range( comp.num_segmentos)) for dim in N)
            #
            # MODEL.add_constraint(landa[p] == MODEL.sum(j*muL[p, j] + gammaL[p, j] for j in range(comp.num_segmentos)))
            # # MODEL.add_constraint(gammaL[p, 0] <= muL[p, 0])
            # # MODEL.add_constraint(gammaL[p, j] <= muL[p, j-1] + muL[p, j] for j in range(1, comp.num_segmentos))
            # # MODEL.add_constraint(gammaL[p, comp.num_segmentos-1] <= muL[p, comp.num_segmentos-1])
            # MODEL.add_constraint(gammaL[p, j] <= muL[p, j] for j in range(comp.num_segmentos))
            # MODEL.add_constraint(pL[p, j] >= muL[p, j] + gammaL[p, j] - 1 for j in range(comp.num_segmentos))
            # MODEL.add_constraint(pL[p, j] <= muL[p, j] for j in range(comp.num_segmentos))
            # MODEL.add_constraint(pL[p, j] <= gammaL[p, j] for j in range(comp.num_segmentos))
            # MODEL.add_constraint(muL.sum(p, '*') == 1)
            # # MODEL.add_constraint(gammaL.sum(p, '*') == 1)
            # MODEL.add_constraint(Lp[p, dim] == MODEL.sum(muL[p, j]*comp.V[j][dim] + pL[p, j] * (comp.V[j+1][dim] - comp.V[j][dim]) for j in range( comp.num_segmentos)) for dim in N)
    
    # MODEL.add_constraint((pgRt.sum('*', t) + MODEL.sum(np.linalg.norm(P[g1-1].V - P[g2-1].V)*yggt[g1, g2, t] for g1 in T_index_prima for g2 in T_index_prima if g1 != g2) + pgLt.sum('*', t))/vD <= 30 for t in T_index_prima)
    
    MODEL.add_constraints(xLt[0, dim] == datos.orig[dim] for dim in N)
    MODEL.add_constraints(xRt[0, dim] == datos.orig[dim] for dim in N)
    #
    MODEL.add_constraints(xLt[datos.m+1, dim] == datos.dest[dim] for dim in N)
    MODEL.add_constraints(xRt[datos.m+1, dim] == datos.dest[dim] for dim in N)
    
    # MODEL.add_constraint(v.sum('*', e, '*') == 1 for e in E_index)
    # MODEL.add_constraint(z.sum(e, '*', '*') == 1 for e in E_index)
        
    # Funcion objetivo
    # + MODEL.sum(0.5*pgLt[index] for index in pgLt.keys()) + MODEL.sum(0.5*pgRt[index] for index in pgRt.keys())
    # objective = MODEL.sum(dRLt[index] for index in dRLt.keys()) + MODEL.sum(dLRt[index] for index in dLRt.keys()) + MODEL.sum(pgLt[index] for index in dgLt.keys()) + MODEL.sum(pgRt[index] for index in dgRt.keys())
    
    objective = MODEL.sum(3*dLRt[index] for index in dLRt.keys()) + MODEL.sum(3*dRLt[index] for index in dRLt.keys()) + MODEL.sum(pgRt[index] for index in pgRt.keys()) + MODEL.sum(pgLt[index] for index in pgLt.keys()) + MODEL.sum(pggt[g1, g2, t] for g1, g2, t in yggt.keys()) + MODEL.sum(pgt[index] for index in pgt.keys())
    
    
    first_time = time.time()
    MODEL.minimize(objective)
    
     
    
    # MODEL.setParam('TimeLimit', datos.tmax)
    
    MODEL.parameters.timelimit = datos.tmax
    timelimit_cb = MODEL.register_callback(TimeLimitCallback)
    timelimit_cb.starttime = time.time()
    timelimit_cb.timeused = 0
    timelimit_cb.objective_value = 0
    
    # MODEL.Params.Threads = 6
    # MODEL.Params.TimeLimit = datos.tmax
    # MODEL.Params.LazyConstraints = 1
    # MODEL._yggt = yggt
    # MODEL._mugt = mugt
    # MODEL._ugt = ugt
    # MODEL._vgt = vgt
    
     
    # MODEL.Params.SolutionLimit = 3
    
    # Optimizamos
    MODEL.solve()
    #MODEL.solve(log_output='solucion'+str(log)+'.log') 
    
    
    

    # MODEL.optimize(subtourelim)
    
    # MODEL.write('aver.lp')
    #
    #
    # if MODEL.Status == 3:
    #     result = []
    #     result =  [np.nan, np.nan, np.nan, np.nan]
    #
    #     MODEL.computeIIS()
    #     MODEL.write('infactible.ilp')
    #
    #     return result
    #
    # if MODEL.SolCount == 0:
    #     result = []
    #     result =  [np.nan, np.nan, np.nan, np.nan]
    #
    #     # MODEL.computeIIS()
    #     # MODEL.write('sinsol.ilp')
    #
    #     return result
    #
    # MODEL.write('./sol_files/multitarget.sol')
    #
    # vals_ugt = MODEL.getAttr('x', ugt)
    #
    # selected_ugt = gp.tuplelist(e for e in vals_ugt if vals_ugt[e] > 0)
    #
    # print('ugt')
    # print(selected_ugt)
    #
    # vals_vgt = MODEL.getAttr('x', vgt)
    #
    # selected_vgt = gp.tuplelist(e for e in vals_vgt if vals_vgt[e] > 0)
    #
    # print('vgt')
    # print(selected_vgt)
    #
    # vals_yggt = MODEL.getAttr('x', yggt)
    #
    # selected_yggt = gp.tuplelist(e for e in vals_yggt if vals_yggt[e] > 0)
    #
    # print('yggt')
    # print(selected_yggt)
    #
    # vals_mugt = MODEL.getAttr('x', mugt)
    #
    # selected_mugt = gp.tuplelist(e for e in vals_mugt if vals_mugt[e] > 0)
    #
    # print('mugt')
    # print(selected_mugt)
    #
    # vals_beta = MODEL.getAttr('x', beta)
    #
    # selected_beta = gp.tuplelist(e for e in vals_beta if vals_beta[e] > 0)
    #
    # print('beta')
    # print(selected_beta)
    #
    #
    # path = []
    # # path.append(0)
    #
    # for t in T_index_prima:
    #     tripleta = selected_ugt.select('*', t)
    #     if tripleta:
    #         path.append(tripleta[0][1])
    #
    # # path.append(nG+1)
    # # print(path)
    #
    # ind = 0
    # path_C = []
    # paths_D = []
    #
    # #path_C.append(orig)
    # path_C.append([xLt[0, 0].X, xLt[0, 1].X])
    # for t in path:
    #     #    if ind < datos.m:
    #     path_C.append([xLt[t, 0].X, xLt[t, 1].X])
    #     path_D = []
    #     path_D.append([xLt[t, 0].X, xLt[t, 1].X])
    #     index_g = g
    #     index_t = t
    #     for g, ti in selected_ugt:
    #         if ti == t:
    #             index_g = g
    #             index_t = ti
    #
    #     # path_D.append([Lgi[index_g, index_i, 0].X, Lgi[index_g, index_i, 1].X])
    #     limite = sum([1 for g, ti in selected_mugt if ti == index_t])
    #     path_D.append([Rp[index_g, 0].X, Rp[index_g, 1].X])
    #     path_D.append([Lp[index_g, 0].X, Lp[index_g, 1].X])
    #     count = 1
    #     while count < limite:
    #         for g1, g2, ti in selected_yggt:
    #             if index_t == ti and index_g == g1:
    #                 count += 1
    #                 index_g = g2
    #                 print(index_g)
    #                 # path_D.append([Rgi[index_g, index_i, 0].X, Rgi[index_g, index_i, 1].X])
    #                 path_D.append([Rp[index_g, 0].X, Rp[index_g, 1].X])
    #                 path_D.append([Lp[index_g, 0].X, Lp[index_g, 1].X])
    #
    #     path_D.append([xRt[t, 0].X, xRt[t, 1].X])
    #     paths_D.append(path_D)
    #     path_C.append([xRt[t, 0].X, xRt[t, 1].X])
    #
    # path_C.append([xLt[datos.m+1, 0].X, xLt[datos.m+1, 1].X])
    #
    # fig, ax = plt.subplots()
    # #
    # # # ax.axis('equal')
    # # ax.set_aspect('equal')
    # # plt.axis([0, 100, 0, 100])
    # #
    # # # plt.axis('equal')
    # #
    # # #
    # # # path_C = []
    # #
    # # plt.plot(xLt[0, 0].X, xLt[0, 1].X, 's', alpha = 1, markersize=10, color='black')
    # # plt.plot(xLt[nG+1, 0].X, xLt[nG+1, 1].X, 's', alpha = 1, markersize=10, color='black')
    # #
    # # for t in path:
    # #     # path_C.append([xLt[t, 0].X, xLt[t, 1].X])
    # #     # path_C.append([xRt[t, 0].X, xRt[t, 1].X])
    # #     # plt.plot(xLt[t, 0].X, xLt[t, 1].X, 's', alpha = 1, markersize=5, color='black')
    # #     # plt.plot(xLt[1, 0].X, xLt[1, 1].X, 'o', alpha = 1, markersize=5,  color='red')
    # #
    # #     # if t == 0:
    # #         # ax.annotate("orig", xy = (xLt[t, 0].X-2, xLt[t, 1].X+1), fontsize = 15)
    # #     if t > 0 and t < nG+1:
    # #         plt.plot(xLt[t, 0].X, xLt[t, 1].X, 'o', alpha = 1, markersize = 5,  color='red')
    # #         plt.plot(xRt[t, 0].X, xRt[t, 1].X, 'o', alpha = 1, markersize = 5,  color='red')
    # #         # ax.annotate("$x_R^{t}$".format(t = t), xy = (xRt[t, 0].X+1.5, xRt[t, 1].X), fontsize = 15)
    # #         # ax.annotate("$x_L^{t}$".format(t = t), xy = (xLt[t, 0].X-3, xLt[t, 1].X), fontsize = 15)
    # #     # if t == nG+1:
    # #     #     plt.plot(xLt[t, 0].X, xLt[t, 1].X, 's', alpha = 1, markersize=10, color='black')
    # #         # ax.annotate("dest", xy = (xLt[t, 0].X+0.5, xLt[t, 1].X+1), fontsize = 15)
    # #
    # # ax.add_artist(Polygon(path_C, fill=False, animated=False, lw = 2, linestyle='-', alpha=1, color='blue'))
    # #
    # # for p in T_index_prima:
    # #     plt.plot(Rp[p, 0].X, Rp[p, 1].X, 'o', markersize=5, color='red')
    # #         # ax.annotate("$R_" + str(g) + "^{" + str((first, second)) + "}$", xy = (Rig[i, g, 0].X+0.75, Rig[i, g, 1].X+0.75))    
    # #     plt.plot(Lp[p, 0].X, Lp[p, 1].X, 'o', markersize=5, color='red')
    # #
    # # for pathd in paths_D:
    # #     ax.add_artist(Polygon(pathd, fill=False, closed=False, lw = 2, animated=False, alpha=1, color='red'))
    # #
    # # for g in T_index_prima:
    # #     ax.add_artist(data[g-1].artist)
    # #

    
    result = []
    
    # if log == 3:
    #     result.append(timelimit_cb.gap)
    #     result.append(timelimit_cb.timeused + heur_time)
    #
    #     return result

    try:
        result.append(MODEL.solve_details.mip_relative_gap)
        result.append(MODEL.solve_details.time)
        result.append(MODEL.solve_details.nb_nodes_processed)
        result.append(MODEL.objective_value)
        
        if datos.init:
            result.append(timelimit_cb.objective_value)
            result.append(timelimit_cb.timeused + heur_time)
    except:
        print('No ha encontrado solucion en el tiempo limite')
        result = [np.nan, np.nan, np.nan, np.nan]
        
        if datos.init:
            result.append(np.nan)
            result.append(np.nan)
            
            return result
    
    
    # infile = r"./solucion" +str(log)+".log"
    #
    # important = []
    # keep_phrases = ['m1']
    #
    # with open(infile) as f:
    #     f = f.readlines()
    #
    # for line in f:
    #     for phrase in keep_phrases:
    #         if phrase in line:
    #             important.append(line)
    #             break
    #
    # print(important)
    # words = important[0].split()
    #
    # starting_obj_val = float(words[-1][0:-1])


    # polygon_camion = Polygon(path_camion, fill=False, linestyle=':', alpha=1, color='red')
    # ax.add_patch(polygon_camion)
    
    
    # plt.title(str(datos.m) + "coordinate ")
    # string = 'imagenes/' + str(m) + "-XPPN - MTZ - Mode " + \
    #     str(modo) + " - Radii - " + str(datos.r) + ".png"
    # plt.savefig('Instance{b}-{d}.png'.format(b = datos.m, d = datos.capacity))
    
    # plt.show()
    
    return result
    
    #plt.close()

    # plt.show()
# MultiTargetPolygonal_CPLEX(datos, 5)
# MultiTargetPolygonal_CPLEX(datos)