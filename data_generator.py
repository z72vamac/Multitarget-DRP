"""
Created on Fri May  21 12:00:00 2021

Generando datos de tamano de 5 a 30, con r = 2, y capacidades de 40 a 80, de 10 en 10.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from data import *
from itertools import combinations
import auxiliar_functions as af
from data import *
from entorno import *
import csv
from entorno import *
import estimacion_M as eM
# import ujson
import json
import pickle
import copy

seed = np.random.seed(2)

instancias = {}

ms = range(5, 30)
capacitys = [40, 50, 60, 70]

tmax = 7200

for m in ms:
    for i in range(5):
        datos = Data([], r = 2, show = True, m = m, modo = 4, init = False, tmax = tmax, capacity = 30, seed = seed)

        datos.generar_muestra()
        instancias[(i, m, 30)] = datos

        for c in capacitys:
            datos2 = copy.copy(datos)
            datos2.capacity = c
            instancias[(i, m, c)] = datos2

with open("instancias_mixtas.pickle","wb") as pickle_out:
    pickle.dump(instancias, pickle_out)

