import numpy as np
import math
import random
from numpy.linalg import inv

import matplotlib.pyplot as plt

nbSamples = 1000

np.matrix([[3 * random.random() + 2, 1] for x in range(nbSamples)])
X = np.matrix([[random.random(), 1] for x in range(nbSamples)])
Y = np.matrix([ 3 * x[0].item(0) + 0.666 for x in X]).transpose()
Gnoise = np.random.normal(0.0, 0.1, len(Y))
Ynoisy = np.matrix([Y[i].item(0) + Gnoise[i] for i in range(len(Y))]).transpose()

# Find a and b
A = inv(X.transpose() * X) * X.transpose() * Y

A = inv(X.transpose() * X) * X.transpose() * Ynoisy
print(A)
#display
#plt.scatter(np.asarray(X[:, 0]), np.asarray(Ynoisy))
