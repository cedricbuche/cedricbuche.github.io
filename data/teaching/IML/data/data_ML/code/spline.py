import numpy as np
import math
import random
from numpy.linalg import inv

import matplotlib.pyplot as plt

nbSamples = 1000

X = np.matrix([[random.random(), 1] for x in range(nbSamples)])
Y = np.matrix([math.log(x[0].item(0)) for x in X]).transpose()

def Iplus(xi, x):
    if x >= xi:
        return x - xi
    else:
        return 0.0

def splinify(xMin, xMax, step, x):
    a = [Iplus(xMin + i * step, x) for i in range(int((xMax - xMin) / step))]
    a.reverse()
    return a + [1]

Xsplines = np.matrix([splinify(0.0, 1.0, 0.05, x[0].item(0)) for x in X])
A = inv(Xsplines.transpose() * Xsplines) * Xsplines.transpose() * Y
Yreg = np.matrix([[ np.dot(x, A).item(0)] for x in Xsplines])

plt.scatter(np.asarray(X[:, 0]), np.asarray(Y))
plt.scatter(np.asarray(X[:, 0]), np.asarray(Yreg))
plt.show()
