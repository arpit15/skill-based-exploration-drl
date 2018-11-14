import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sys import argv
import os.path as osp
from scipy import signal
import seaborn as sns
from collections import OrderedDict

NUM_RUNS = 5

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])



# use lab-v1 for putaoutb
# use cluster-v1 for other envs

# 10 for picknmove only
env_name = argv[1]
env_name = env_name.split('-')

extensions = ['', 'stone', 'hie','paramnoise', 'flat']
color_list = ['g', 'r', 'y', 'b', 'm']
# agent_name_list = ['Baseline 1', 'Baseline 2', 'Approach 1', 'Approach 2']
agent_name_list = ['HER','PAS','HER-PAS', 'HER+ParamNoise','HER+Lookahead(Our)']
b, a = signal.butter(1, 0.15)

for agent_name, ext, color in zip(agent_name_list, extensions, color_list):
	data = []

	if ext is 'stone':
		currlinestyle = 'loosely dotted'
	elif ext is 'hie':
		currlinestyle = 'dashdotdotted'
	elif ext is 'paramnoise':
		currlinestyle = 'dotted'
	elif ext is 'flat':
		currlinestyle = 'densely dashdotdotted'
	else:
		currlinestyle = 'solid'

	linewidth = 1.0
	# print(ext)
	if ext is 'paramnoise':

		curr_data = pd.read_csv("/Users/virtualworld/new_RL3/corl_paper_results/lab-v1/%s%s-%s/run%d/progress.csv"%(env_name[0], ext, env_name[1], 1)).fillna(0.)["eval/success"]
		# print(curr_data.shape)
		filtered_data = signal.filtfilt(b,a, curr_data)
		data.append(filtered_data*100)
		# plt.plot(filtered_data*100, linestyle= linestyles[currlinestyle])
		# continue
		linewidth = 3.0
	else:
		for j in range(1,NUM_RUNS+1):
			curr_data = pd.read_csv("/Users/virtualworld/new_RL3/corl_paper_results/lab-v1/%s%s-%s/run%d/progress.csv"%(env_name[0], ext, env_name[1], j)).fillna(0.)["eval/success"]
			# print(curr_data.shape)
			filtered_data = signal.filtfilt(b,a, curr_data)
			data.append(filtered_data*100)

	ax = sns.tsplot(data=data, color=color, ci="sd", linestyle= linestyles[currlinestyle], linewidth = linewidth)
	print("tsplot %s"%ext)

plt.legend(agent_name_list)
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Success %')
plt.show()