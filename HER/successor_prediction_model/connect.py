import numpy as np
from os.path import expanduser

# reacher
pos_data = np.loadtxt(expanduser("~/pred_models/Reacher3d-v0/run1/Reacher3d-v0_pos.csv"), delimiter=',')
neg_data = np.loadtxt('/tmp/suc/Reacher3d-v0_neg.csv', delimiter=',')
labels = np.zeros((neg_data.shape[0],1))
neg_data = np.concatenate((neg_data, labels), axis=1)

print(pos_data.shape, neg_data.shape)
reacher_data = np.concatenate((pos_data, neg_data), axis=0)
np.savetxt(expanduser("~/pred_models/Reacher3d-v0/run1/Reacher3d-v0_data.csv"), reacher_data, delimiter=',')

# grasping
pos_data = np.loadtxt(expanduser("~/pred_models/grasping-v2/run2/grasping-v2_pos.csv"), delimiter=',')
neg_data = np.loadtxt('/tmp/suc/grasping-v2_neg.csv',  delimiter=',')

labels = np.zeros((neg_data.shape[0],1))
neg_data = np.concatenate((neg_data, labels), axis=1)
grasping_data = np.concatenate((pos_data, neg_data), axis=0)
np.savetxt(expanduser("~/pred_models/grasping-v2/run2/grasping-v2_data.csv"), grasping_data, delimiter=',')
