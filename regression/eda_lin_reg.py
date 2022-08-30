import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import seaborn as sns

from sklearn.model_selection import train_test_split

path = os.getcwd()
std_scaler = StandardScaler()

data1 = path + "\data\\airfoil_self_noise.dat"
col1 = ['Freq', 'Angle', 'Chord', 'Velocity', 'Thickness', 'Sound'] # columns of the dataset
df1 = pd.read_table(data1, sep="\t", names=col1)

# df1 = df1[(df1['Freq']<=10000) & (df1['Thickness']<=0.05)]


## correlation matrix
hm = sns.heatmap(df1[['Freq', 'Angle', 'Chord', 'Velocity', 'Thickness']].corr(), annot = True)
hm.set(title = "Correlation Matrix\n")
plt.show()

sns.set_theme(style="whitegrid")
## data distribution
sns.displot(df1, x="Freq")
sns.displot(df1, x="Angle")
sns.displot(df1, x="Chord")
sns.displot(df1, x="Velocity")
sns.displot(df1, x="Thickness")
sns.displot(df1, x="Sound")
plt.show()

# print(df1[(df1['Freq']<=10000) & (df1['Thickness']<=0.05)].shape)
outliers = [y for stat in boxplot_stats(df1['Angle']) for y in stat['fliers']]
print(len(outliers))
# outliers.sort()
print(outliers)

sns.boxplot(x=df1["Freq"], width=0.5)
plt.show()
# sns.boxplot(x=df1["Angle"], width=0.5)
# plt.show()
# sns.boxplot(x=df1["Chord"], width=0.5)
# plt.show()
# sns.boxplot(x=df1["Velocity"], width=0.5)
# plt.show()
sns.boxplot(x=df1["Thickness"], width=0.5)
plt.show()