from scipy import spatial
import numpy as np
from sys import argv


env_name = argv[1]
csv_filename = '/tmp/her/%s.csv' % env_name

data = np.loadtxt(csv_filename, delimiter=',')
tree = spatial.KDTree(data[:,:6])

results = tree.query(data[:, :6],k=2)[1][:,1]
s0 = data[-3:][results]

delta = data[-3:] - s0

delta_dataset = np.concatenate((data[:,:6], s0, delta))

np.save('/tmp/her/%s-delta.csv'%env_name)

