import numpy as np
from os.path import expanduser

# transfer
pos_data = np.loadtxt(expanduser("~/pred_models/transfer-v0/run3/transfer-v0_pos.csv"), delimiter=',')
neg_data = np.loadtxt('/tmp/suc/grasping-v2_neg.csv',  delimiter=',')

labels = np.zeros((neg_data.shape[0],1))
neg_data = np.concatenate((neg_data, labels), axis=1)
grasping_data = np.concatenate((pos_data, neg_data), axis=0)
np.savetxt(expanduser("~/pred_models/transfer-v0/run3/transfer-v0_data.csv"), grasping_data, delimiter=',')
