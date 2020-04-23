import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# regular_data = pd.read_csv("Invader_baseline.csv")
# sv_data = pd.read_csv("Invader_sva2c.csv")
# regular_data = pd.read_csv("Qbert_baseline.csv")
# sv_data = pd.read_csv("Qbert_sva2c.csv")
# regular_data = pd.read_csv("Assault_baseline.csv")
# sv_data = pd.read_csv("Assault_sva2c_0.085.csv")
# regular_data = pd.read_csv("Pong_baseline.csv")
# sv_data = pd.read_csv("Pong_sva2c_0.093.csv")
# regular_data = pd.read_csv("Seaquset_baseline.csv")
# sv_data = pd.read_csv("Seaquest_sva2c_0.085.csv")
# regular_data = pd.read_csv("Breakout_baseline.csv")
# sv_data = pd.read_csv("Breakout_sva2c_0.09.csv")
regular_data = pd.read_csv("AirRaid_baseline.csv")
sv_data = pd.read_csv("AirRaid_sva2c_0.09.csv")

len = sv_data.shape[0]
print(len)

data = pd.DataFrame(np.ones((len, 4)))
data.columns = ['step', 'avg_reward', 'algorithm', 'seed']

i = 0
all = pd.DataFrame([])

for row in regular_data.iteritems():
    data['step, 0.09'] = np.linspace(0, len-1, len).reshape(-1, 1)
    col_1 = regular_data.iloc[:,i]
    data['avg_reward'] = np.reshape(np.array(col_1.values), (-1, 1))
    data['algorithm'] = 'regular'
    data['seed'] = i
    i += 1
    all = pd.concat([all, data], 0)
i = 0
for row in sv_data.iteritems():
    data['step, 0.09'] = np.linspace(0, len-1, len).reshape(-1, 1)
    col_2 = sv_data.iloc[:,i]
    data['avg_reward'] = np.reshape(np.array(col_2.values), (-1, 1))
    data['algorithm'] = 'sv_a2c'
    data['seed'] = i
    i += 1
    all = pd.concat([all, data], 0)

plt.figure()
sns.lineplot(x='step, 0.09', y='avg_reward', data=all, hue='algorithm')
plt.savefig('AirRaid_0.09.png', format="png")
  
