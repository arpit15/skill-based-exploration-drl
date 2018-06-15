import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sys import argv
import os.path as osp

env_name = argv[1]
env_name = env_name.split('-')
ddpgher = pd.read_csv("/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%s-%s/run1/progress.csv"%(env_name[0], env_name[1])).fillna(0.)
paramher = pd.read_csv("/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%shie-%s/run1/progress.csv"%(env_name[0], env_name[1])).fillna(0.)
lookahead = pd.read_csv("/Users/virtualworld/new_RL3/corl_paper_results/clusters-v1/%sflat-%s/run1/progress.csv"%(env_name[0], env_name[1])).fillna(0.)

plt.plot(list(range(200)), ddpgher["eval/success"])
plt.plot(list(range(200)), paramher["eval/success"])
plt.plot(list(range(200)), lookahead["eval/success"])

plt.show()