import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import seaborn as sns
from logistic_reg_func import similar

path = os.getcwd()
std_scaler = StandardScaler()

data1 = path + "\data\\data_banknote_authentication.txt"
col1 = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']
df1 = pd.read_table(data1, sep=",", names=col1)

## correlation matrix
# hm = sns.heatmap(df1[['Variance', 'Skewness', 'Curtosis', 'Entropy']].corr(), annot = True)
# hm.set(xlabel='\nIRIS Flower Details', ylabel='IRIS Flower Details\t', title = "Correlation matrix of IRIS data\n")
# hm.set(title = "Correlation matrix\n")
# plt.show()

# sns.set_theme(style="whitegrid")
## data distribution
# sns.displot(tips, x="size", discrete=True)
# sns.displot(penguins, x="flipper_length_mm", col="sex", bins=30)
# sns.displot(df1, x="Variance")
# sns.displot(df1, x="Skewness")
# sns.displot(df1, x="Curtosis")
# sns.displot(df1, x="Entropy")
# sns.displot(df1, x="Class", discrete=True)
# plt.show()

# sns.boxplot(x=df1["Variance"], width=0.5)
# plt.show()
# sns.boxplot(x=df1["Skewness"], width=0.5)
# plt.show()
# sns.boxplot(x=df1["Curtosis"], width=0.5)
# plt.show()
# sns.boxplot(x=df1["Entropy"], width=0.5)
# plt.show()

# print(df1[(df1['Freq']<=10000) & (df1['Thickness']<=0.05)].shape)
# outliers = [y for stat in boxplot_stats(df1['Entropy']) for y in stat['fliers']]
# print(len(outliers))
# outliers.sort()
# print(outliers)

# measure similarity score
other = [ -0.002 ,-0.02 , -0.014 , 0.003 , 0.005]
baseline = [-1.568, -4.928, -5.041, -4.615,  0.237]
print( round( similar(other,baseline),3) ) 