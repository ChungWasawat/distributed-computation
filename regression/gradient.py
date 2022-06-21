import os

from numpy.random import seed
from numpy.random import rand
from numpy.random import shuffle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

path = os.getcwd()
#############################################################################
""""""
data1 = path + "\data\\accelerometer.csv"
col1 = ""

# df1 = pd.read_csv(data)
# X = pd.DataFrame(df1['x'])
# y = pd.DataFrame(df1['wconfid'])

##############################################################################
"""
Data2 has the following inputs:
1. Frequency, in Hertzs.
2. Angle of attack, in degrees.
3. Chord length, in meters.
4. Free-stream velocity, in meters per second.
5. Suction side displacement thickness, in meters.

The only output is:
6. Scaled sound pressure level, in decibels.
"""
data2 = path + "\data\\airfoil_self_noise.dat"
col2 = ['freq', 'angle', 'chord', 'velocity', 'thickness', 'sound']

df1 = pd.read_table(data2, sep="\t", names=col2)
X = df1.values[:, 0::3]
y = df1.values[:, 5]

model = LinearRegression()
model.fit(X, y)
print(model.intercept_, model.coef_)

###### stochastic gradient descent
seed(47)

gradient = rand(X.shape[1])
learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 1]

print(shuffle( np.array( [i for i in range(10)]) ) )

for lr in learning_rate:
    pass

