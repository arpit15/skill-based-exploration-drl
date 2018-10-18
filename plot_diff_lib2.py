from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from sys import argv

b, a = signal.butter(1, 0.15)

agent_name_list = ['HER', 'HER+Skill Set 1','HER+Skill Set 2']

# putinb
env_name = argv[1]
env_name = env_name.split('-')

dir_list = ['/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%s-%s/run1'%(env_name[0], env_name[1]),
			'/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%sflat-%s/run1'%(env_name[0], env_name[1]),
			'/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%sflat-%s/run16'%(env_name[0], env_name[1])]
for dirname in dir_list:
	data = pd.read_csv(join(dirname , "progress.csv")).fillna(0.0)

	filtered_data = signal.filtfilt(b,a, data["eval/success"])
	plt.plot(data["total/epochs"], filtered_data)

plt.legend(agent_name_list)
plt.xlabel('Number of epochs')
plt.ylabel('Success %')
plt.show()