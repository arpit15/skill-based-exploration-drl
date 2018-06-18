import os.path as osp
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from HER.common.mpi_moments import mpi_moments


def read_checkpoint_local(restore_dir):
    if(osp.exists(osp.join(restore_dir, 'checkpoint'))):
        with open(osp.join(restore_dir, 'checkpoint'), 'r') as f:
            firstline = f.readline().split('"')[1]
            # get the basename 
            model_name = osp.basename(firstline)
            # check existence
            filename = osp.join(restore_dir,model_name)
            if(osp.exists(filename + '.meta')):
                return filename
            else:
                return None 

    else:
        print('No checkpoint file found!')
        return None


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def mpi_mean(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0][0]


def mpi_std(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[1][0]


def mpi_max(value):
    global_max = np.zeros(1, dtype='float64')
    local_max = np.max(value).astype('float64')
    MPI.COMM_WORLD.Reduce(local_max, global_max, op=MPI.MAX)
    return global_max[0]


def mpi_sum(value):
    global_sum = np.zeros(1, dtype='float64')
    local_sum = np.sum(np.array(value)).astype('float64')
    MPI.COMM_WORLD.Reduce(local_sum, global_sum, op=MPI.SUM)
    return global_sum[0]

def normal_mean(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return np.mean(np.array(value))


def normal_std(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return np.std(np.array(value))
