import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sys import argv
import os.path as osp
from scipy import signal

env_name = argv[1]
env_name = env_name.split('-')
ddpgher = pd.read_csv("/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%s-%s/run1/progress.csv"%(env_name[0], env_name[1])).fillna(0.)
paramher = pd.read_csv("/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%shie-%s/run1/progress.csv"%(env_name[0], env_name[1])).fillna(0.)
lookahead = pd.read_csv("/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%sflat-%s/run1/progress.csv"%(env_name[0], env_name[1])).fillna(0.)
stone = pd.read_csv("/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%sstone-%s/run1/progress.csv"%(env_name[0], env_name[1])).fillna(0.)

# filter
b, a = signal.butter(3, 0.05)
ddpgher_succ = signal.filtfilt(b, a, ddpgher["eval/success"])
paramher_succ = signal.filtfilt(b, a, paramher["eval/success"])
lookahead_succ = signal.filtfilt(b, a, lookahead["eval/success"])
stone_succ = signal.filtfilt(b, a, stone["eval/success"])


plt.plot(list(range(200)), ddpgher_succ)
plt.plot(list(range(200)), paramher_succ)
plt.plot(list(range(200)), lookahead_succ)
plt.plot(list(range(200)), stone_succ)

plt.show()