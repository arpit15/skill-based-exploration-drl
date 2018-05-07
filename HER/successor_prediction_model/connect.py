import numpy as np
from os import path, expanduser

# reacher
pos_data = np.loadtxt(expanduser("~/pred_model/Reacher3d-v0/run1/Reacher3d-v0_pos.csv"))
neg_data = np.loadtxt('/tmp/suc/Reacher3d-v0_pos.csv')
reacher_data = np.concatenate((pos_data, neg_data), axis=0)
np.savetxt(expanduser("~/pred_model/Reacher3d-v0/run1/Reacher3d-v0_data.csv"))

# grasping
pos_data = np.loadtxt(expanduser("~/pred_model/grasping-v2/run2/grasping-v2_pos.csv"))
neg_data = np.loadtxt('/tmp/suc/grasping-v2_pos.csv')
reacher_data = np.concatenate((pos_data, neg_data), axis=0)
np.savetxt(expanduser("~/pred_model/grasping-v2/run2/grasping-v2_data.csv"))
