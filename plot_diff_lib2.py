from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from sys import argv
from collections import OrderedDict
b, a = signal.butter(1, 0.15)

agent_name_list = ['HER', 'HER+Skill Set 1','HER+Skill Set 2']

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



# putinb
# env_name = argv[1]
env_name = "picknmove-v2"
env_name = env_name.split('-')

dir_list = ['/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%s-%s/run11'%(env_name[0], env_name[1]),
			'/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%sflat-%s/run1'%(env_name[0], env_name[1]),
			'/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%sflat-%s/run16'%(env_name[0], env_name[1])]
for i, dirname in enumerate(dir_list):
	data = pd.read_csv(join(dirname , "progress.csv")).fillna(0.0)

	if i == 0:
		currlinestyle = 'densely dotted'
	elif i ==2:
		currlinestyle = 'dashdotted'
	else:
		currlinestyle = 'densely dashdotdotted'

	filtered_data = signal.filtfilt(b,a, data["eval/success"])
	plt.plot(data["total/epochs"], filtered_data, linewidth = 3.0, linestyle = linestyles[currlinestyle])

plt.legend(agent_name_list)
plt.xlabel('Number of epochs')
plt.ylabel('Success %')
plt.show()