from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from sys import argv

b, a = signal.butter(1, 0.15)

agent_name_list = ['DDPG+HER', 'DDPG+HER+SL1']

# 
dir_list = ['/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/putainb-v1/run21',
			'/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/putainbflat-v1/run21']


# dir_list = ['/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/putainb-v2/run31',
# 			'/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/putainbflat-v2/run31']

# big wall
# dir_list = ['/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/putainb-v3/run51',
# 			'/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/putainbflat-v3/run51']

for dirname in dir_list:
	data = pd.read_csv(join(dirname , "progress.csv")).fillna(0.0)

	filtered_data = signal.filtfilt(b,a, data["eval/success"])
	plt.plot(data["total/epochs"], filtered_data)

plt.legend(agent_name_list)
plt.xlabel('Number of epochs')
plt.ylabel('Success %')
plt.show()