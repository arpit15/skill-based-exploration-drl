from scipy import spatial
import numpy as np
from sys import argv


env_name = argv[1]
shape = int(argv[2])

csv_filename = '/tmp/her/%s.csv' % env_name

data = np.loadtxt(csv_filename, delimiter=',')
tree = spatial.KDTree(data[:,:shape])

results = tree.query(data[:, :shape],k=2)[1][:,1]
s0 = data[:,shape:][results]

delta = data[:, shape:] - s0

print(data[:,:shape].shape, s0.shape, delta.shape)

delta_dataset = np.concatenate((data[:,:shape], s0, delta),axis=1)

np.savetxt('/tmp/her/%s-delta.csv'%env_name, delta_dataset, delimiter=',')

