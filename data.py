# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 08:31:09 2020

En este documento generamos los datos según las configuraciones detalladas
en el documento xpp_segmentos2.

@author: Carlos
"""

# Paquetes
import numpy as np
import entorno as e
import matplotlib.pyplot as plt
import random
import math
from copy import copy


class Data(object):

    def __init__(self, data, m, r, capacity, modo, tmax, init, show, vC = 1, vD = 3, orig = [50, 50], dest = [50, 50], seed=0):
        self.data = data
        self.m = m
        self.r = r
        self.capacity = capacity
        self.modo = modo
        self.tmax = tmax
        # self.alpha = alpha
        self.init = init
        self.show = show
        self.vC = vC
        self.vD = vD
        self.orig = orig
        self.dest = dest
        self.grid_list = []
        random.seed(seed)

    def generar_elipse(self):
        radio = 5*np.random.uniform(self.r-1, self.r)
        centro = np.random.uniform(radio, 100-radio, 2)
        # centro = np.random.uniform(0, 100, 2)
        P = np.identity(2)
        q = -2*centro
        r = centro[0]**2 + centro[1]**2 - radio**2
        elipse = e.Elipse(P, q, r)
        self.data.append(elipse)

    def generar_poligono(self):
        radio = 5*np.random.uniform(self.r-1, self.r)
        centro = np.random.uniform(radio, 100-radio, 2)
        nV = np.random.randint(3, 10)
        angulos = np.linspace(0, 2*np.pi, num=nV+1) + 360*np.random.rand()
        V = np.stack((centro[0] + radio*np.cos(angulos),
                     centro[1] + radio*np.sin(angulos))).T
        self.data.append(e.Poligono(V))

    def generar_poligonal(self):
        # genero radio en funcion del tamaño
        radio = 5*np.random.uniform(self.r-1, self.r)
        V = []
        # genero un centro en el cuadrado [0, 100]
        P = np.random.uniform(radio, 100-radio, 2)
        nV = 4
#        nV = 9
        angulo = 360*np.random.rand()  # genero un ángulo aleatorio de giro del segmento
        V.append(P)
        for i in range(nV-1):
            P = V[i]
            angulo += 30
            Q = np.array([P[0] + radio*np.cos(angulo),
                         P[1] + radio*np.sin(angulo)])
            while Q[0] <= radio or Q[1] <= radio or Q[0] >= 100-radio or Q[1] >= 100-radio:
                #                angulo = 360*np.random.rand()
                angulo += 30
                Q = np.array([P[0] + radio*np.cos(angulo),
                             P[1] + radio*np.sin(angulo)])
            V.append(Q)
        alpha = np.random.rand()
        self.data.append(e.Poligonal(V, alpha))

    def generar_punto(self):
        P = np.random.uniform(0, 100, 2)
        self.data.append(e.Punto(P))
        
    def generar_muestra(self):
        if self.modo == 1:
            for i in range(self.m):
                self.generar_punto()
        if self.modo == 2:
            for i in range(self.m):
                self.generar_poligonal()
        if self.modo == 3:
            for i in range(self.m):
                flag = np.random.randint(1, 3)
                if flag == 1:
                    self.generar_punto()
                elif flag == 2:
                    self.generar_poligonal()
        if self.modo == 4:
            m2 = int(math.floor(self.m/2))
            for i in range(m2):
                self.generar_punto()

            for i in range(self.m-m2):
                self.generar_poligonal()

    def vaciar_muestra(self):
        self.olddata = copy(self.data)
        self.data = []

    def elimina_entornos(self, lista):
        self.data = np.delete(self.data, lista)
        self.m = self.m - len(lista)

    def reduce_radio(self, porcentaje):
        datos = copy(self.data)
        self.data = []
        if self.modo == 1:
            for entorno in datos:
                P = entorno.P
                q = entorno.q
                radio = entorno.radio*(1 - porcentaje)
                r = entorno.centro[0]**2 + entorno.centro[1]**2 - radio**2
                self.data.append(e.Elipse(P, q, r))

    def recupera_radio(self):
        self.data = copy(self.olddata)
        self.olddata = []

    def mostrar_datos(self):
        return self.data

    def imprimir_datos(self):
        for i in self.data:
            print(i)

    def cambiar_init(self):
        if self.init:
            self.init = False
        else:
            self.init = True

    def dibujar_muestra(self):
        if self.init:
            fig = plt.figure()
            ax2 = fig.add_subplot(111)

            min_x = []
            max_x = []
            min_y = []
            max_y = []

            for c in range(self.m):
                dato = self.data[c]
                if type(dato) is e.Elipse:
                    min_x.append(dato.centro[0] - dato.width)
                    max_x.append(dato.centro[0] + dato.width)
                    min_y.append(dato.centro[1] - dato.height)
                    max_y.append(dato.centro[1] + dato.height)
                    ax2.annotate(s=str(c), xy=(dato.centro[0], dato.centro[1]))
                if type(dato) is e.Poligono:
                    min_x.append(min(P[0] for P in dato.V))
                    max_x.append(max(P[0] for P in dato.V))
                    min_y.append(min(P[1] for P in dato.V))
                    max_y.append(max(P[1] for P in dato.V))
                    ax2.annotate(s=str(c), xy=(
                        dato.baricentro[0], dato.baricentro[1]))

                ax2.add_artist(dato.artist)
                # dato = self.olddata[c]
                # if type(dato) is e.Elipse:
                #     min_x.append(dato.centro[0] - dato.width)
                #     max_x.append(dato.centro[0] + dato.width)
                #     min_y.append(dato.centro[1] - dato.height)
                #     max_y.append(dato.centro[1] + dato.height)
                #     ax2.annotate(s = str(c), xy=(dato.centro[0], dato.centro[1]))
                # if type(dato) is e.Poligonal or type(dato) is e.Poligono:
                #     min_x.append(min(P[0] for P in dato.V))
                #     max_x.append(max(P[0] for P in dato.V))
                #     min_y.append(min(P[1] for P in dato.V))
                #     max_y.append(max(P[1] for P in dato.V))
                #     ax2.annotate(s = str(c), xy=(dato.baricentro[0], dato.baricentro[1]))
                # ax2.add_artist(dato.artist)

            #ax2.autoscale_view()
            ax2.axis([0, 100, 0, 100])
            #ax2.axis([min(min_x)-1, max(max_x)+1, min(min_y)-1, max(max_y)+1])
            ax2.set_aspect('equal')
            self.init = False

        if not(self.init):
            fig = plt.gcf()

        return fig
