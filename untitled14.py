# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:40:22 2020

@author: atohidi
"""
import pandas as pd
import numpy as np


filePath = 'C:\\Users\\atohidi\\Downloads\\Description (2).txt'


from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aux = pd.read_csv(filePath, sep=';')  # Pandas dataframe
data = aux.to_numpy()  # data[row,column] for each element with numpy
aux = aux[['U_act', 'Hyp', 'TimeGS']].round(6)

# print(aux['TimeGS'][(aux['U_act'] == 0.316667) & (aux['Hyp'] == 60)].values[0])

def f(x, y):
    #print(aux['TimeGS'][(aux['U_act'] == x) & (aux['Hyp'] == y)].values[0])
    val = aux['TimeGS'][(aux['U_act'] == x) & (aux['Hyp'] == y)]
    if len(val) != 0:
        return val.values[0]
    else:
        return 0

f = np.vectorize(f)

x = data[:, 4]
y = data[:, 5]


X, Y = np.meshgrid(x, y)
Z = f(X, Y)
print(Z)
fig = plt.figure()

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
             cmap='viridis', edgecolor='none');
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
