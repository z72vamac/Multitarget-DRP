"""Tenemos un conjunto E de entornos ugt un conjunto de poligonales P de las que queremos recorrer un porcentaje alfa p . Buscamos
   un tour de mínima distancia que alterne poligonal-entorno ugt que visite todas las poligonales"""


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
from auxiliar_functions import path2matrix

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

np.random.seed(129)

datos = Data([], m = 4,
                 r = 2,
                 modo = 3,
                 tmax = 1200,
                 init = 0,
                 show = True,
                 seed = 2)
datos.generar_muestra()

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
m = 10

P = np.random.uniform(0, 100, (m, 2))

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

ugt = MODEL.addVars(ugt_index, vtype=GRB.BINARY, name='y1')


# Variable binaria ugt = 1 si se entra por el grafo g en la etapa t

mugt_index = ugt_index

mugt = MODEL.addVars(mugt_index, vtype=GRB.BINARY, name='mu')


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

MODEL.update()

if datos.init:
    MODEL.read('./sol_files/multitarget.sol')

MODEL.addConstrs(mugt[g, t] - ugt[g, t] == yggt.sum('*', g, t) for g, t in ugt.keys())
MODEL.addConstrs(mugt[g, t] - vgt[g, t] == yggt.sum(g, '*', t) for g, t in ugt.keys())

MODEL.addConstrs(mugt.sum(g, '*') == 1 for g in T_index_prima)

# for t in T_index_prima:
#     MODEL.addConstrs(yggt[g1, g2, t] + yggt[g2, g1, t] <= 1 for g1 in T_index_prima for g2 in T_index_prima if g1 != g2)
# for g in T_index_prima:
#     for t1 in T_index_prima:
#         MODEL.addConstr(gp.quicksum(mugt[g, t2] for t2 in T_index_prima if t2 < t1) <= mugt[g, t1])
# En cada etapa hay que visitar/salir un segmento de un grafo
# for t in T_index_prima:
#     MODEL.addConstrs(ugt[g1, t] + ugt[g2, t] <= 1 for g1 in T_index_prima for g2 in T_index_prima if g1 != g2)
#     MODEL.addConstrs(vgt[g1, t] + vgt[g2, t] <= 1 for g1 in T_index_prima for g2 in T_index_prima if g1 != g2)


MODEL.addConstrs(ugt.sum('*', t) <= 1 for t in T_index_prima)
MODEL.addConstrs(vgt.sum('*', t) <= 1 for t in T_index_prima)

# for g in T_index_prima:
#     for t in T_index_prima:
#         MODEL.addConstr(sgt[g, t] <= mugt.sum('*', t) - 1)
#         MODEL.addConstr(sgt[g, t] >= 0)
#
# # Eliminación de subtours
# MODEL.addConstrs(mugt.sum('*', t) - 1 >= (sgt[g1, t] - sgt[g2, t]) + grafos[g-1].num_aristas * yggt[g1, g2, t] for g1, g2, t in yggt.keys())


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))

for t in T_index_prima:
    restricciones = MODEL.addConstrs(gp.quicksum(yggt[g1, g2, t] for g1, g2 in permutations(set, 2)) <= len(set) - 1 for set in list(powerset(T_index_prima)))
    restricciones.Lazy = 3


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


MODEL.addConstrs((difgLt[g, t, dim] >=  xLt[t, dim] - P[g-1, dim]) for g, t, dim in difgLt.keys())
MODEL.addConstrs((difgLt[g, t, dim] >= -xLt[t, dim] + P[g-1, dim]) for g, t, dim in difgLt.keys())

MODEL.addConstrs(difgLt[g, t, 0]*difgLt[g, t, 0] + difgLt[g, t, 1]*difgLt[g, t, 1] <= dgLt[g, t]*dgLt[g, t] for g, t in dgLt.keys())

SmallM = 0
BigM = max([np.linalg.norm(np.array(P[q]) - np.array(P[p])) for p in range(datos.m) for q in range(datos.m)])
# BigM = 10000

MODEL.addConstrs(pgLt[g, t] >= SmallM * ugt[g, t] for g, t in pgLt.keys())
MODEL.addConstrs(pgLt[g, t] >= dgLt[g, t] - BigM * (1 - ugt[g, t]) for g, t in pgLt.keys())

MODEL.addConstr(gp.quicksum(BigM*(1-ugt[g, t]) - dgLt[g, t] for g, t in ugt.keys()) <= gp.quicksum(dgLt[g, t] for g, t in ugt.keys()))
MODEL.addConstr(gp.quicksum(BigM*(1-ugt[g, t]) - dgLt[g, t] for g, t in ugt.keys()) <= gp.quicksum(BigM*ugt[g, t] for g, t in ugt.keys()))

MODEL.addConstr(gp.quicksum(BigM*(1-vgt[g, t]) - dgRt[g, t] for g, t in ugt.keys()) <= gp.quicksum(dgRt[g, t] for g, t in ugt.keys()))
MODEL.addConstr(gp.quicksum(BigM*(1-vgt[g, t]) - dgRt[g, t] for g, t in ugt.keys()) <= gp.quicksum(BigM*vgt[g, t] for g, t in ugt.keys()))


# MODEL.addConstrs(BigM*(1 - ugt[g, t]) <= dgLt[g, t] for g, t in pgLt.keys())
# MODEL.addConstrs(BigM*(1 - vgt[g, t]) <= dgRt[g, t] for g, t in pgLt.keys())


MODEL.addConstrs((difgRt[g, t, dim] >=  P[g-1, dim] - xRt[t, dim]) for g, t, dim in difgRt.keys())
MODEL.addConstrs((difgRt[g, t, dim] >= -P[g-1, dim] + xRt[t, dim]) for g, t, dim in difgRt.keys())

MODEL.addConstrs(difgRt[g, t, 0]*difgRt[g, t, 0] + difgRt[g, t, 1]*difgRt[g, t, 1] <= dgRt[g, t]*dgRt[g, t] for g, t in dgRt.keys())

MODEL.addConstrs(pgRt[g, t] >= SmallM * vgt[g, t] for g, t in pgRt.keys())
MODEL.addConstrs(pgRt[g, t] >= dgRt[g, t] - BigM * (1 - vgt[g, t]) for g, t in pgRt.keys())

MODEL.addConstrs((difRLt[t, dim] >=   xRt[t, dim] - xLt[t + 1, dim]) for t in T_index_primaprima for dim in range(2))
MODEL.addConstrs((difRLt[t, dim] >=  -xRt[t, dim] + xLt[t + 1, dim]) for t in T_index_primaprima for dim in range(2))
MODEL.addConstrs(difRLt[t, 0]*difRLt[t, 0] + difRLt[t, 1] * difRLt[t, 1] <= dRLt[t] * dRLt[t] for t in T_index_primaprima)

MODEL.addConstrs((difLRt[t, dim] >=   xLt[t, dim] - xRt[t, dim]) for t, dim in difLRt.keys())
MODEL.addConstrs((difLRt[t, dim] >= - xLt[t, dim] + xRt[t, dim]) for t, dim in difLRt.keys())
MODEL.addConstrs(difLRt[t, 0]*difLRt[t, 0] + difLRt[t, 1] * difLRt[t, 1] <= dLRt[t] * dLRt[t] for t in dLRt.keys())

MODEL.addConstrs((pgRt.sum('*', t) + gp.quicksum(np.linalg.norm(P[g1-1] - P[g2-1])*yggt[g1, g2, t] for g1 in T_index_prima for g2 in T_index_prima if g1 != g2) + pgLt.sum('*', t))/vD <= dLRt[t]/vC for t in T_index_prima)

MODEL.addConstrs((dLRt[t]/vC <= 200) for t in dLRt.keys())

# MODEL.addConstrs((pgRt.sum('*', t) + gp.quicksum(np.linalg.norm(P[g1-1].V - P[g2-1].V)*yggt[g1, g2, t] for g1 in T_index_prima for g2 in T_index_prima if g1 != g2) + pgLt.sum('*', t))/vD <= 30 for t in T_index_prima)

MODEL.addConstrs(xLt[0, dim] == datos.orig[dim] for dim in N)
MODEL.addConstrs(xRt[0, dim] == datos.orig[dim] for dim in N)
#
MODEL.addConstrs(xLt[datos.m+1, dim] == datos.dest[dim] for dim in N)
MODEL.addConstrs(xRt[datos.m+1, dim] == datos.dest[dim] for dim in N)

# MODEL.addConstrs(v.sum('*', e, '*') == 1 for e in E_index)
# MODEL.addConstrs(z.sum(e, '*', '*') == 1 for e in E_index)

MODEL.update()

# Funcion objetivo
# + gp.quicksum(0.5*pgLt[index] for index in pgLt.keys()) + gp.quicksum(0.5*pgRt[index] for index in pgRt.keys())
# objective = gp.quicksum(dRLt[index] for index in dRLt.keys()) + gp.quicksum(dLRt[index] for index in dLRt.keys()) + gp.quicksum(pgLt[index] for index in dgLt.keys()) + gp.quicksum(pgRt[index] for index in dgRt.keys())

objective = gp.quicksum(3*dLRt[index] for index in dLRt.keys()) + gp.quicksum(3*dRLt[index] for index in dRLt.keys()) + gp.quicksum(pgRt[index] for index in pgRt.keys()) + gp.quicksum(pgLt[index] for index in pgLt.keys()) + gp.quicksum(np.linalg.norm(P[p-1] - P[q-1])*yggt[p, q, t] for p, q, t in yggt_index)

MODEL.setObjective(objective, GRB.MINIMIZE)

MODEL.update()

MODEL.setParam('TimeLimit', datos.tmax)
MODEL.write('aver.lp')

MODEL.Params.NumericFocus = 0

# Optimizamos
MODEL.optimize()

if MODEL.Status == 3:

    MODEL.computeIIS()
    MODEL.write('infactible.ilp')

MODEL.write('./sol_files/multitarget.sol')

vals_ugt = MODEL.getAttr('x', ugt)

selected_ugt = gp.tuplelist(e for e in vals_ugt if vals_ugt[e] > 0)

print('ugt')
print(selected_ugt)

vals_vgt = MODEL.getAttr('x', vgt)

selected_vgt = gp.tuplelist(e for e in vals_vgt if vals_vgt[e] > 0)

print('vgt')
print(selected_vgt)

vals_yggt = MODEL.getAttr('x', yggt)

selected_yggt = gp.tuplelist(e for e in vals_yggt if vals_yggt[e] > 0)

print('yggt')
print(selected_yggt)

vals_mugt = MODEL.getAttr('x', mugt)

selected_mugt = gp.tuplelist(e for e in vals_mugt if vals_mugt[e] > 0)

print('mugt')
print(selected_mugt)



path = []
# path.append(0)

for t in T_index_prima:
    tripleta = selected_ugt.select('*', t)
    if tripleta:
        path.append(tripleta[0][1])

# path.append(nG+1)
print(path)

ind = 0
path_C = []
paths_D = []

#path_C.append(orig)
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

    count = 0
    path_D.append([P[index_g-1, 0], P[index_g-1, 1]])
    # path_D.append([Lgi[index_g, index_i, 0].X, Lgi[index_g, index_i, 1].X])
    limite = sum([1 for g1, g2, ti in selected_yggt if ti == index_t])
    while count < limite:
        for g1, g2, ti in selected_yggt:
            if index_t == ti and index_g == g1:
                count += 1
                index_g = g2
                # path_D.append([Rgi[index_g, index_i, 0].X, Rgi[index_g, index_i, 1].X])
                path_D.append([P[index_g-1, 0], P[index_g-1, 1]])

    path_D.append([xRt[t, 0].X, xRt[t, 1].X])
    paths_D.append(path_D)
    path_C.append([xRt[t, 0].X, xRt[t, 1].X])

path_C.append([xLt[datos.m+1, 0].X, xLt[datos.m+1, 1].X])

fig, ax = plt.subplots()

plt.axis([0, 100, 0, 100])

#
# path_C = []
for t in path:
    # path_C.append([xLt[t, 0].X, xLt[t, 1].X])
    # path_C.append([xRt[t, 0].X, xRt[t, 1].X])
    plt.plot(xLt[t, 0].X, xLt[t, 1].X, 'ko', alpha = 0.3, markersize=10, color='green')
    ax.annotate("L" + str(t), xy = (xLt[t, 0].X, xLt[t, 1].X))
    plt.plot(xRt[t, 0].X, xRt[t, 1].X, 'ko', markersize=5, color='blue')
    ax.annotate("R" + str(t), xy = (xRt[t, 0].X, xRt[t, 1].X))

ax.add_artist(Polygon(path_C, fill=False, animated=False,
              linestyle='-', alpha=1, color='blue'))

for pathd in paths_D:
    ax.add_artist(Polygon(pathd, fill=False, closed=False, lw = 1,
                  animated=False, alpha=1, color='red'))

for g in T_index_prima:
    plt.plot(P[g-1, 0], P[g-1, 1], 'ko', markersize=5, color='black')

# polygon_camion = Polygon(path_camion, fill=False, linestyle=':', alpha=1, color='red')
# ax.add_patch(polygon_camion)


plt.title(str(datos.m) + "coordinate ")
# string = 'imagenes/' + str(m) + "-XPPN - MTZ - Mode " + \
#     str(modo) + " - Radii - " + str(datos.r) + ".png"
plt.savefig('poikonen.png')
#plt.show()
#plt.close()
plt.show()
