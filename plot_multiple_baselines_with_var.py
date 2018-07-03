import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sys import argv
import os.path as osp
from scipy import signal
import seaborn as sns

NUM_RUNS = 5

env_name = argv[1]
env_name = env_name.split('-')

extensions = ['', 'stone', 'hie', 'flat']
color_list = ['g', 'r', 'y', 'b']
# agent_name_list = ['Baseline 1', 'Baseline 2', 'Approach 1', 'Approach 2']
agent_name_list = ['DDPG+HER','PAS MDP','MPAS MDP','DDPG+HER+SL']
b, a = signal.butter(1, 0.15)

for agent_name, ext, color in zip(agent_name_list, extensions, color_list):
	data = []
	for j in range(1,NUM_RUNS+1):
		curr_data = pd.read_csv("/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%s%s-%s/run%d/progress.csv"%(env_name[0], ext, env_name[1], j+10)).fillna(0.)["eval/success"]
		# print(curr_data.shape)
		filtered_data = signal.filtfilt(b,a, curr_data)
		data.append(filtered_data)

	ax = sns.tsplot(data=data, value=agent_name,color=color)

plt.legend(agent_name_list)
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Success %')
plt.show()