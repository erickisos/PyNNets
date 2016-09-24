#!/usr/bin/python2
# -*- encoding:utf-8 -*-

"""
    Conjunto de utilidades para el trabajo con redes neuronales, Work In
    Progress, posteriormente se añadirá una interfaz gráfica y un control
    independiente para trabajar con algoritmos evolutivos.
"""

import random
import math
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname) - 10s] :> %(message)s')


class Neuron(object):
    """
    Clase Neurona, base para la creación de redes neuronales, entrenamientos
    independientes WIP
    """

    def __init__(self, name=0, inputsNum=1, isinput=True):
        """
        En su iniciación se crean los pesos aleatorios para un número de
        entradas dado
        :param name: (uint) Es el índice de la neurona en caso de pertenecer a
        una red, default = 0
        :param inputsNum: (uint) Es el númeo total de entradas que tendrá la
        neurona, por default se añade una entrada más que representa al Bias.
        """
        self._name = name
        self._isinput = isinput
        if not self._isinput:
            self._inputsNumber = inputsNum + 1
        else:
            self._inputsNumber = inputsNum
        self._w = None
        self.resetWeights()

    def setInputs(self, val):
        """
        Funcion que establece el número total de entradas que tendrá la neurona
        se reevalua el número de entradas y se resetean los pesos globales
        :param val: (int) Número de entradas
        """
        self._inputsNumber = val
        self.resetWeights()

    @staticmethod
    def _weights_generator(n):
        """
        Método estático que toma un número "n" de entradas y devuelve una lista
        que contiene el valor de los pesos generados aleatoriamente
        :param n: (uint) Número de entradas, debe ser un entero sin signo
        :return: (list) lista de pesos (vector) generado de manera aleatoria
        """
        weights = [random.uniform(-1, 1) for i in range(n)]
        return weights

    def getWeights(self):
        """
        Función que devuelve el vector de pesos de la neurona para evitar la
        necesidad de acceder directamente a la variable W
        :return: (list) Vector de pesos propio de la neurona
        """
        return self._w

    def resetWeights(self):
        """
        Función que reinicia los valores de los pesos a números aleatorios en
        el espacio de soluciones
        """
        self._w = self._weights_generator(self._inputsNumber)

    def addInput(self, n):
        self._w = self._weights_generator(self._inputsNumber + n)

    def reIndex(self, name=0):
        self._name = name

    def getIndex(self):
        return self._name

    def feedForward(self, val, fact="sigmo"):
        if fact == "sigmo":
            func = lambda x: (1.0 / (1 - math.exp(-x)))
            diff = lambda x: (x(1.0 - x))
        elif fact == "tanh":
            func = lambda x: math.tanh(x)
            diff = lambda x: (1.0 - x ** 2)
        else:
            func = lambda x: 1 if (x > 0) else 0
        val = list(val)
        if not self._isinput:
            val.append(1)
        y = sum([self._w[i] * float(val[i]) for i in range(len(val))])
        if self._isinput:
            return y
        y = func(y)
        return y

    def setWeights(self, new_w):
        self._w = new_w

    def backPropagation(self, f_act='sigmo'):
        if f_act == 'tanh':
            diff = lambda x: (1 - (x ** 2))
        elif f_act == 'sigmo':
            diff = lambda x: (x * (1 - x))

    def getOutput(self):
        return self._output

    def __str__(self):
        return str(self._name)


class NeuralManager(object):
    """
        Clase de control que permite la estructuración libre de redes neurales
        basándonos en conexiones recursivas o lineales indiferentemente,
        primordialmente útil en el uso de algoritmos genéticos evolutivos.
    """
    def __init__(self):
        logging.info("Creando nuevo Manager")
        self._neuralList = dict()   # -- Dict donde se almacenarán las neuronas
        self._neuralConnections = list()
        self._neuralIndex = 0
        self._neuralCurrentOutputs = dict()
        self._neuralLastOutputs = dict()
        logging.info("Listo, a la espera de comandos")

    def updateOutputs(self):
        self._neuralLastOutputs = dict(self._neuralCurrentOutputs)
        for key in self._neuralList:
            self._neuralCurrentOutputs[key] = self._neuralList[key].getOutput()

    def addNeuron(self):
        self._neuralList[self._neuralIndex] = Neuron(name=self._neuralIndex,
                                                     inputsNum=0)
        self._neuralIndex += 1
        logging.info("Neurona añadida con éxito!")

    def delNeuron(self, neuralIndex=0):
        if neuralIndex in self._neuralList:
            logging.debug("Eliminando neurona")
            self._neuralList.pop(neuralIndex)
            self.delConnections(neuralIndex)
        else:
            logging.warn("La neurona no existe!")

    def addConnection(self, neuralOne=0, neuralTwo=0):
        if (neuralOne, neuralTwo) not in self._neuralConnections:
            self._neuralConnections.append((neuralOne, neuralTwo))

    def delConnections(self, neuralIndex=0):
        pass


class NeuralNet(object):
    def __init__(self, cbk, topology=(1,)):
        self._cbk = cbk
        self._inputNeurons = len(self._cbk[0][0])
        self._outputNeurons = len(self._cbk[0][1])
        self._topology = list(topology)
        # Validating Topology
        if self._topology[0] == self._inputNeurons:
            print("Mismo numero")
            if self._topology[-1] == self._outputNeurons:
                print("Todo bien hasta aquí")
            else:
                print("Podemos forzar la red...")
        else:
            print("Red recursiva acaso?")
        self._net = self.makeTheNet()
        # Initializing Salidas
        self._outputs = list()
        for capa in range(len(self._topology)):
            capa_list = list()
            for elem in range(self._topology[capa]):
                capa_list.append(elem)
            self._outputs.append(capa_list)

        self._deltas = list()
        for capa in range(len(self._topology) - 1):
            capa_list = list()
            for elem in range(self._topology[capa + 1]):
                capa_list.append(elem)
            self._deltas.append(capa_list)
        self._errors = [0.0] * 2

    def feedForward(self, inputs):
        counter = 0
        for i in self._net:
            print("vamos bien")
            neurona = 0
            for j in i:
                print("Salida actual: {}".format(
                    self._outputs[counter][neurona]))
                if counter == 0:
                    lista = []
                    lista.append(inputs[neurona])
                    self._outputs[counter][neurona] = j.feedForward(
                            lista)
                else:
                    self._outputs[counter][neurona] = j.feedForward(
                            self._outputs[counter - 1])
                print("Nueva salida actual: {}".format(
                    self._outputs[counter][neurona]))
                neurona += 1
                print("Numero de Neurona: {}".format(neurona))
            print("Capa: {}".format(counter))
            counter += 1
        return self._outputs[len(self._topology) - 1][:]

    def makeTheNet(self):
        net = list()
        counter = 0
        for i in range(len(self._topology)):
            net.append(list())
            for j in range(self._topology[i]):
                if i == 0:
                    net[i].append(Neuron(name=counter, inputsNum=1,
                                         isinput=True))
                    net[i][j].setWeights([1])
                else:
                    net[i].append(Neuron(name=counter,
                                         inputsNum=len(net[i - 1]),
                                         isinput=False))
                counter += 1
        return net

    def backPropagation(self, targets, eta=0.6, alpha=0.8):
        for i in range(-1, -len(self._topology), -1):
            error = targets[i] - self._outputs[i]

    def resetWeights(self, n=0):
        pass

    def __str__(self):
        return str(self._topology)
