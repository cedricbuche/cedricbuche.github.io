import numpy as np
import math
import random
from numpy.linalg import inv
import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt

cols = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance']

df = pd.read_csv('JFKraw.csv', usecols=cols)

#236 manhattan upper east side
JFK_MU = df[(df['PULocationID'] == 132) & (df['DOLocationID'] == 236)]

#JFK_MU = JFK_MU[(JFK_MU['trip_distance'] > 18.3) & (JFK_MU['trip_distance'] < 20.3)]
JFK_MU['weekday'] = JFK_MU['tpep_pickup_datetime'].apply(lambda x : parser.parse(x).weekday())

JFK_MU = JFK_MU[JFK_MU['weekday'] < 5]

pu = [parser.parse(dt) for dt in JFK_MU['tpep_pickup_datetime'].values]
do = [parser.parse(dt) for dt in JFK_MU['tpep_dropoff_datetime'].values]
dur = [(b -a).total_seconds() / 3600.0 for a, b in zip(pu, do)]
startTime = [dt.hour + dt.minute / 60.0 for dt in pu]


X = startTime
Y = dur

def Iplus(xi, x):
    if x >= xi:
        return x - xi
    else:
        return 0.0

def splinify(xMin, xMax, step, x):
    a = [Iplus(xMin + i * step, x) for i in range(int((xMax - xMin) / step))]
    a.reverse()
    return a + [1]


Xm = np.matrix([[Iplus(0.5, x), Iplus(0, x), 1] for x in X])

Gnoise = np.random.normal(0.0, 0.1, len(Y))
Ynoisy = np.matrix([Y[i] + Gnoise[i] for i in range(len(Y))]).transpose()

# Find a and b
Xm = np.matrix([splinify(np.min(X), np.max(X), 2., x) for x in X])
A = inv(Xm.transpose() * Xm) * Xm.transpose() * np.matrix(Y).transpose()
Yreg = np.matrix([[ np.dot(x, A).item(0)] for x in Xm])

plt.xlabel('Heure Depart (h)')
plt.ylabel('Duree Trajet (h)')

plt.scatter(X, np.asarray(Y))
plt.scatter(X, np.asarray(Yreg))
plt.show()
