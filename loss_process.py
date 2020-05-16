import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

filePath = './baseline.csv'
file2Path = './sva2c.csv'
alpha = 0.1
init_len = 3
with open(filePath, encoding='utf-8') as f:
    data = np.loadtxt(f, float, delimiter=',', usecols=2)

with open(file2Path, encoding='utf-8') as f:
    data2 = np.loadtxt(f, float, delimiter=',', usecols=2)


def calnxt(s):
    s2 = np.zeros(s.shape)
    s2[0] = s[0]
    for i in range(1, len(s2)):
        s2[i] = alpha * s[i] + (1 - alpha) * s2[i - 1]
    return s2

def caltriple(datas):
    init_value = float(datas[:init_len].sum()) / init_len
    init_value = np.array([init_value])
    for c in datas:
        init_value = np.append(init_value, c)
    s_single = calnxt(init_value)
    s_double = calnxt(s_single)
    s_triple = calnxt(s_double)
    return s_triple

def getdf(datas, algorithm):
    pda = caltriple(datas[0:333])
    pdb = caltriple(datas[333:666])
    pdc = caltriple(datas[666:])

    dfa = pd.DataFrame(np.array(pda[1:]))
    dfb = pd.DataFrame(np.array(pdb[1:]))
    dfc = pd.DataFrame(np.array(pdc[1:len(pdc) - 1]))

    dfa['step'] = np.array([i for i in range(1, len(pda))])
    dfb['step'] = np.array([i for i in range(1, len(pdb))])
    dfc['step'] = np.array([i for i in range(1, len(pdc) - 1)])

    dfa.columns = ['avg_bellman_loss', 'step']
    dfb.columns = ['avg_bellman_loss', 'step']
    dfc.columns = ['avg_bellman_loss', 'step']
    df = pd.concat([dfa, dfb, dfc], 0)
    df['algorithm'] = algorithm
    return df

df1 = getdf(data, 'baseline')
df2 = getdf(data2, 'sv_a2c')
df = pd.concat([df1, df2], 0)

ax = sns.lineplot(x='step', y='avg_bellman_loss', data=df, hue='algorithm')
plt.show()
