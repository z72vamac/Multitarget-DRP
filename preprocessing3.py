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

# np.random.seed(3)
# #
# # m = 10
#
# datos = Data([], m = 6,
#                  r = 2,
#                  modo = 3,
#                  tmax = 60,
#                  init = 0,
#                  show = True,
#                  seed = 2)
# datos.generar_muestra()

def preprocessing3(datos, number):

    conjunto = []

    data = datos.mostrar_datos()

    n = 2
    # datos.orig = [0, 0]
    # datos.dest = [0, 0]
    vD = datos.vD
    vC = datos.vC
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

    subconjuntos = combinations(T_index_prima, number)

    for lista in subconjuntos:
        
        print(lista)
        
        lista = list(lista)

        # Creacion del modelo
        MODEL = gp.Model('klmedian-continuous')


        # Variable binaria ugt = 1 si se entra por el grafo g en la etapa t

        ugt_index = []

        for g in lista:
            ugt_index.append(g)

        ugt = MODEL.addVars(ugt_index, vtype=GRB.BINARY, name='ugt')


        # Variable binaria ugt = 1 si se entra por el grafo g en la etapa t

        mugt_index = ugt_index

        mugt = MODEL.addVars(mugt_index, vtype=GRB.BINARY, name='mugt')
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


        for g1 in lista:
            for g2 in lista:
                if g1 != g2:
                    yggt_index.append((g1, g2))

        # print(yggt_index)
        
        yggt = MODEL.addVars(yggt_index, vtype = GRB.BINARY, name = 'yggt')
        dggt = MODEL.addVars(yggt_index, vtype = GRB.CONTINUOUS, name = 'Dggt')
        difggt = MODEL.addVars(yggt_index, 2, vtype = GRB.CONTINUOUS, name = 'difggt')
        pggt = MODEL.addVars(yggt_index, vtype = GRB.CONTINUOUS, name = 'pggt')

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

        # # Variable continua no negativa dRLt que indica la distancia entre el punto de recogida en la etapa t ugt el punto de
        # # salida para la etapa t+1
        # dRLt_index = T_index_primaprima
        #
        # dRLt = MODEL.addVars(dRLt_index, vtype=GRB.CONTINUOUS, lb=0.0, name='dRLt')
        # difRLt = MODEL.addVars(dRLt_index, 2, vtype=GRB.CONTINUOUS, lb=0.0, name='difRLt')

        # Variable continua no negativa dLRt que indica la distancia que recorre el camión en la etapa t mientras el dron se mueve

        dLRt = MODEL.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='dLRt')
        difLRt = MODEL.addVars(2, vtype=GRB.CONTINUOUS, lb=0.0, name='difLRt')

        # Variables que modelan los puntos de entrada o recogida
        # xLt: punto de salida del dron del camion en la etapa t
        xLt_index = []

        for dim in range(2):
            xLt_index.append(dim)

        xLt = MODEL.addVars(xLt_index, vtype=GRB.CONTINUOUS, name='xLt')

        # xRt: punto de recogida del dron del camion en la etapa t
        xRt_index = []

        for dim in range(2):
            xRt_index.append(dim)

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

        for p in lista:
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
                
            for dim in range(2):
                Rp_index.append((p, dim))
                Lp_index.append((p, dim))


        landa = MODEL.addVars(landa_index, vtype=GRB.CONTINUOUS, name='landa')
        rho = MODEL.addVars(rho_index, vtype=GRB.CONTINUOUS, name='rho')

        gammaR = MODEL.addVars(gammaR_index, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='gammaR')
        gammaL = MODEL.addVars(gammaL_index, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='gammaL')

        muR = MODEL.addVars(muR_index, vtype=GRB.BINARY, name='muR')
        muL = MODEL.addVars(muL_index, vtype=GRB.BINARY, name='muL')

        pR = MODEL.addVars(muR_index, vtype=GRB.BINARY, name='pR')
        pL = MODEL.addVars(muR_index, vtype=GRB.BINARY, name='pL')

        u = MODEL.addVars(u_index, vtype=GRB.BINARY, name='u')
        dp = MODEL.addVars(lista, vtype = GRB.CONTINUOUS, name = 'dp')

        landa_min = MODEL.addVars(u_index, vtype=GRB.CONTINUOUS, lb=0.0, name='landa_min')
        landa_max = MODEL.addVars(u_index, vtype=GRB.CONTINUOUS, lb=0.0, name='landa_max')

        Rp = MODEL.addVars(Rp_index, vtype = GRB.CONTINUOUS, name = 'Rp')
        Lp = MODEL.addVars(Lp_index, vtype = GRB.CONTINUOUS, name = 'Lp')

        beta = MODEL.addVars(T_index_prima, vtype = GRB.BINARY, name = 'beta')

        MODEL.update()

        # if datos.init:
        #     MODEL.read('./sol_files/multitarget.sol')

        MODEL.addConstrs(mugt[g] - ugt[g] == yggt.sum('*', g) for g in ugt.keys())
        MODEL.addConstrs(mugt[g] - vgt[g] == yggt.sum(g, '*') for g in ugt.keys())

        # MODEL.addConstrs(mugt.sum(g, '*') == 1 for g in T_index_prima)

        # for t in T_index_prima:
        #     MODEL.addConstrs(yggt[g1, g2] + yggt[g2, g1, t] <= 1 for g1 in T_index_prima for g2 in T_index_prima if g1 != g2)
        # for g in T_index_prima:
        #     for t1 in T_index_prima:
        #         MODEL.addConstr(gp.quicksum(mugt[g2] for t2 in T_index_prima if t2 > t1) <= gp.quicksum(mugt[g2] for t2 in T_index_prima if t1 <= t2))

        # for t in T_index_prima:
        #     for m in T_index_prima:
        #         print(datos.m)
        #         print(m)
        #         MODEL.addConstr(gp.quicksum(mugt[g, datos.m - j + 1] for j in T_index_prima for g in T_index_prima if j < m and datos.m - j < t) <= m - mugt.sum(g))

        # for t in T_index_prima:
        #     MODEL.addConstr(datos.m*(1 - beta[t]) >= datos.m - gp.quicksum(mugt[g1] for g in T_index_prima for t1 in T_index_prima if t1 <= t))
            # MODEL.addConstr(gp.quicksum(mugt[g] for g in T_index_prima for t1 in T_index_prima if t1 > t) <= datos.m*(1 - beta[t]))

        # for t in T_index_prima[0:-1]:
        #     MODEL.addConstr(beta[t] <= beta[t+1])

        # for t in T_index_prima:
        #     MODEL.addConstr(mugt.sum(g) >= 1 - beta[t])
            # MODEL.addConstr(gp.quicksum(mugt[g1] for g in T_index_prima for t1 in T_index_prima if t1 <= t) - gp.quicksum(mugt[g1] for g in T_index_prima for t1 in T_index_prima if t1 <= t-1) >= 1)
        # for t in T_index_prima:
        #     MODEL.addConstr(gp.quicksum(mugt[g1] for g in T_index_prima for t1 in T_index_prima if t1 > t) <= datos.m + gp.quicksum(mugt[g1] for g in T_index_prima for t1 in T_index_prima if t1 <= t))
        # En cada etapa hay que visitar/salir un segmento de un grafo
        # for t in T_index_prima:
        #     MODEL.addConstrs(ugt[g1, t] + ugt[g2, t] <= 1 for g1 in T_index_prima for g2 in T_index_prima if g1 != g2)
        #     MODEL.addConstrs(vgt[g1, t] + vgt[g2, t] <= 1 for g1 in T_index_prima for g2 in T_index_prima if g1 != g2)


        MODEL.addConstr(ugt.sum('*') == 1)
        MODEL.addConstr(vgt.sum('*') == 1)

        # for g in T_index_prima:
        #     for t in T_index_prima:
        #         MODEL.addConstr(sgt[g] <= mugt.sum(g) - 1)
        #         MODEL.addConstr(sgt[g] >= 0)
        #
        # # Eliminación de subtours
        # MODEL.addConstrs(mugt.sum(g) - 1 >= (sgt[g1, t] - sgt[g2, t]) + grafos[g-1].num_aristas * yggt[g1, g2] for g1, g2 in yggt.keys())


        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))

        restricciones = MODEL.addConstrs(gp.quicksum(yggt[g1, g2] for g1, g2 in permutations(set, 2)) <= len(set) - 1 for set in list(powerset(lista)))
        restricciones.Lazy = 2

        MODEL.addConstrs(difggt[g1, g2, dim] >=  Lp[g1, dim] - Rp[g2, dim] for g1, g2, dim in difggt.keys())
        MODEL.addConstrs(difggt[g1, g2, dim] >= -Lp[g1, dim] + Rp[g2, dim] for g1, g2, dim in difggt.keys())
        MODEL.addConstrs(gp.quicksum(difggt[g1, g2, dim]*difggt[g1, g2, dim] for dim in N) <= dggt[g1, g2] * dggt[g1, g2] for g1, g2 in dggt.keys())

        # BigM = 10000
        # SmallM = 0

        for g1, g2 in yggt.keys():

            comp1 = data[g1-1]
            comp2 = data[g2-1]

            SmallM = min_dist(comp1, comp2)[0]
            # SmallM = min([np.linalg.norm(u - v) for u in comp1.V for v in comp2.V])
            BigM = max([np.linalg.norm(u - v) for u in comp1.V for v in comp2.V])

            MODEL.addConstr(pggt[g1, g2] >= SmallM * yggt[g1, g2])
            MODEL.addConstr(pggt[g1, g2] >= dggt[g1, g2] - BigM * (1 - yggt[g1, g2]))


        # MODEL.addConstrs(ugt[g] <= yggt.sum(g, '*') for g in ugt.keys())
        # MODEL.addConstrs(vgt[g] <= yggt.sum('*', g) for g in vgt.keys())


        # MODEL.addConstrs(1 - ugt.sum(g, '*') == yggt.sum(g, '*') for g in T_index_prima)
        # MODEL.addConstrs(1 - vgt.sum(g, '*') == yggt.sum('*', g) for g in T_index_prima)


        # MODEL.addConstrs(ugt.sum(g) <= 1 for t in T_index_prima)
        # MODEL.addConstrs(vgt.sum(g) <= 1 for t in T_index_prima)
        #
        # MODEL.addConstrs(ugt[g] + yggt.sum('*', g) == vgt[g] + yggt.sum(g, g) for g in ugt.keys())
        #
        # MODEL.addConstrs(ugt.sum(g) == vgt.sum(g) for t in T_index_prima)


        MODEL.addConstrs((difgLt[g, dim] >=  xLt[dim] - Rp[g, dim]) for g, dim in difgLt.keys())
        MODEL.addConstrs((difgLt[g, dim] >= -xLt[dim] + Rp[g, dim]) for g, dim in difgLt.keys())

        MODEL.addConstrs(difgLt[g, 0]*difgLt[g, 0] + difgLt[g, 1]*difgLt[g, 1] <= dgLt[g]*dgLt[g] for g in dgLt.keys())

        SmallM = 0
        # BigM = max([np.linalg.norm(np.array(comp[p].V) - np.array(P[p].V)) for p in range(datos.m) for q in range(datos.m)])
        BigM = 10000

        MODEL.addConstrs(pgLt[g] >= SmallM * ugt[g] for g in pgLt.keys())
        MODEL.addConstrs(pgLt[g] >= dgLt[g] - BigM * (1 - ugt[g]) for g in pgLt.keys())

        # MODEL.addConstr(gp.quicksum(BigM*(1-ugt[g]) - dgLt[g] for g in ugt.keys()) <= gp.quicksum(dgLt[g] for g in ugt.keys()))
        # MODEL.addConstr(gp.quicksum(BigM*(1-ugt[g]) - dgLt[g] for g in ugt.keys()) <= gp.quicksum(BigM*ugt[g] for g in ugt.keys()))
        #
        # MODEL.addConstr(gp.quicksum(BigM*(1-vgt[g]) - dgRt[g] for g in ugt.keys()) <= gp.quicksum(dgRt[g] for g in ugt.keys()))
        # MODEL.addConstr(gp.quicksum(BigM*(1-vgt[g]) - dgRt[g] for g in ugt.keys()) <= gp.quicksum(BigM*vgt[g] for g in ugt.keys()))


        # MODEL.addConstrs(BigM*(1 - ugt[g]) <= dgLt[g] for g in pgLt.keys())
        # MODEL.addConstrs(BigM*(1 - vgt[g]) <= dgRt[g] for g in pgLt.keys())


        MODEL.addConstrs((difgRt[g, dim] >=   Lp[g, dim] - xRt[dim]) for g, dim in difgRt.keys())
        MODEL.addConstrs((difgRt[g, dim] >=  -Lp[g, dim] + xRt[dim]) for g, dim in difgRt.keys())

        MODEL.addConstrs(difgRt[g, 0]*difgRt[g, 0] + difgRt[g, 1]*difgRt[g, 1] <= dgRt[g]*dgRt[g] for g in dgRt.keys())

        MODEL.addConstrs(pgRt[g] >= SmallM * vgt[g] for g in pgRt.keys())
        MODEL.addConstrs(pgRt[g] >= dgRt[g] - BigM * (1 - vgt[g]) for g in pgRt.keys())

        # MODEL.addConstrs((difRLt[dim] >=   xRt[dim] - xLt[dim]) for t in T_index_primaprima for dim in range(2))
        # MODEL.addConstrs((difRLt[dim] >=  -xRt[dim] + xLt[dim]) for t in T_index_primaprima for dim in range(2))
        # MODEL.addConstrs(difRLt[t, 0]*difRLt[t, 0] + difRLt[t, 1] * difRLt[t, 1] <= dRLt[t] * dRLt[t] for t in T_index_primaprima)

        # MODEL.addConstrs((difLRt[dim] >=   xLt[dim] - xRt[dim]) for dim in difLRt.keys())
        # MODEL.addConstrs((difLRt[dim] >= - xLt[dim] + xRt[dim]) for dim in difLRt.keys())
        # MODEL.addConstrs(difLRt[t, 0]*difLRt[t, 0] + difLRt[t, 1] * difLRt[t, 1] <= dLRt[t] * dLRt[t] for t in dLRt.keys())

        MODEL.addConstrs(pgt[g] >= SmallM * mugt[g] for g in mugt.keys())
        MODEL.addConstrs(pgt[g] >= dp[g] - BigM * (1 - mugt[g]) for g in mugt.keys())


        MODEL.addConstr((pgLt.sum('*') + pgt.sum('*') + gp.quicksum(pggt[g1, g2] for g1, g2 in pggt.keys()) + pgRt.sum('*'))/vD <= dLRt/vC)
        #
        # MODEL.addConstrs((pgLt.sum(g) + pgt.sum(g) + gp.quicksum(pggt[g1, g2] for g1 in T_index_prima for g2 in T_index_prima if g1 != g2) + pgRt.sum(g))/vD <= dLRt[t]/vC + BigM*(1 - ugt.sum(g)) for t in T_index_prima)

        MODEL.addConstr(dLRt/vC <= datos.capacity)

        MODEL.addConstrs(mugt[g] == 1 for g in lista)

        for p in lista:
            comp = data[p-1]
            if type(comp) is Poligonal:
                MODEL.addConstr(rho[p] - landa[p] == landa_max[p] - landa_min[p], name='u0')
                # si u = 0, entonces landa0 >= landa1
                MODEL.addConstr(landa_max[p] + landa_min[p] >= comp.alpha * comp.num_segmentos, name='u1')
                MODEL.addConstr(landa_max[p] <= comp.num_segmentos * (1 - u[p]), name='u2')
                MODEL.addConstr(landa_min[p] <= comp.num_segmentos * u[p], name='u3')
                MODEL.addConstr(dp[p] == (landa_max[p] + landa_min[p])*comp.longitud/comp.num_segmentos)
        
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
                    MODEL.addConstr(dp[p] == 0)
        # MODEL.addConstrs((pgRt.sum(g) + gp.quicksum(np.linalg.norm(P[g1-1].V - P[g2-1].V)*yggt[g1, g2] for g1 in T_index_prima for g2 in T_index_prima if g1 != g2) + pgLt.sum(g))/vD <= 30 for t in T_index_prima)

        # MODEL.addConstrs(xLt[0, dim] == datos.orig[dim] for dim in N)
        # MODEL.addConstrs(xRt[0, dim] == datos.orig[dim] for dim in N)
        #
        # MODEL.addConstrs(xLt[datos.m+1, dim] == datos.dest[dim] for dim in N)
        # MODEL.addConstrs(xRt[datos.m+1, dim] == datos.dest[dim] for dim in N)

        # MODEL.addConstrs(v.sum('*', e, '*') == 1 for e in E_index)
        # MODEL.addConstrs(z.sum(e, '*', '*') == 1 for e in E_index)

        MODEL.update()

        # Funcion objetivo
        # + gp.quicksum(0.5*pgLt[index] for index in pgLt.keys()) + gp.quicksum(0.5*pgRt[index] for index in pgRt.keys())
        # objective = gp.quicksum(dRLt[index] for index in dRLt.keys()) + gp.quicksum(dLRt[index] for index in dLRt.keys()) + gp.quicksum(pgLt[index] for index in dgLt.keys()) + gp.quicksum(pgRt[index] for index in dgRt.keys())

        objective = gp.quicksum(mugt[index] for index in mugt.keys())

        MODEL.setObjective(objective, GRB.MAXIMIZE)

        MODEL.update()

        # MODEL.setParam('TimeLimit', datos.tmax)
        # MODEL.write('aver.lp')

        MODEL.Params.Threads = 6
        MODEL.Params.OutputFlag = 0
        # MODEL.Params.SolutionLimit = 3

        # Optimizamos
        MODEL.optimize()

        if MODEL.Status == 3:

            # MODEL.computeIIS()
            # MODEL.write('infactible.ilp')

            print('Desigualdad valida encontrada!')

            conjunto.append(lista)

    return conjunto

# conjunto = preprocessing3(datos, 3)
# print(conjunto)
