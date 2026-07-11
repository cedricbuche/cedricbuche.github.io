import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np

df = pd.read_csv('players_stats.csv')


height = df.dropna()['Height']
weight = df.dropna()['Weight']

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')


X = np.zeros((len(height), 2))
X[:,0] = height
X[:,1] = 1
Xm = np.matrix(X)
#Y = np.matrix(weight.as_matrix()) #as_matrix method is deprecated since 0.23.0
Y = np.matrix(weight.to_numpy())
A = inv(Xm.transpose() * Xm) * Xm.transpose() * Y.transpose()
print(A)

x = [160, 230]
y = [[x[0], 1], [x[1], 1]] * A

plt.scatter(height, weight)
plt.plot(x, y, color='r')
plt.show()
