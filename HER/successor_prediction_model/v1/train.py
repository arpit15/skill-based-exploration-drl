import argparse
import os
import gym
import csv
import os.path as osp
import tensorflow as tf
import numpy as np 
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from HER.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from HER.successor_prediction_model.models import classifier
import HER.envs
from HER.ddpg.skills import DDPGSkill
import HER.common.tf_util as U

def run(env_id, render, log_dir, restore_dir, commit_for, 
            train_epoch, batch_size=32, lr = 1e-3, seed = 0, dataset_size=2000):
    
    env = gym.make(env_id)
    observation_shape = env.observation_space.shape[-1]
    global in_size, out_size
    in_size = observation_shape
    out_size = 1

    set_global_seeds(seed)
    env.seed(seed)

    
    with U.single_threaded_session() as sess:
        
        pred_model = classifier(in_shape = in_size, 
                            out_shape = out_size,
                            name = "suc_pred_model", sess=sess,
                            log_dir=log_dir)
        


        init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer(), 
                               )
        sess.run(init_op)

        ## creating dataset tensors
        csv_filename = osp.join(log_dir, "%s.csv"%env_id)

        ## 
        base_dataset = np.loadtxt(csv_filename, delimiter=',')
        train, test = train_test_split(base_dataset, test_size=0.2)
        train_feat = train[:-1]
        train_labels = train[-1]
        # print(train.shape, test.shape)

        # whiten
        train_feat_mean = np.mean(train_feat, axis=0)
        train_feat_std = np.std(train_feat, axis=0)

        # save mean and var
        statistics = np.concatenate((train_feat_mean, train_feat_std))
        with open(osp.join(log_dir, "%s_stat.npy"%env_id), 'wb') as f:
            np.save(f, statistics)

        # create pd
        train_feat_dataset = ( ( train_feat - train_feat_mean)/train_feat_std)
        train_dataset = pd.DataFrame(np.concatenate((train_feat_dataset, train_labels)))
        
        test_feat_dataset = ( ( test[:-1] - train_feat_mean)/train_feat_std)
        test_dataset = [test_feat_dataset, test[-1]]
        ####

        print(train_dataset.shape, test_dataset[0].shape)
        pred_model.train(train_epoch, batch_size, lr, train_dataset , test_dataset)
        pred_model.save()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Baxter-v1')
    boolean_flag(parser, 'render', default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--log-dir', type=str, default='/tmp/her')
    parser.add_argument('--restore-dir', type=str, default=None)

    parser.add_argument('--dataset-size', type=int, default=2000)
    # parser.add_argument('--skillset', type=str, default='set8')
    # parser.add_argument('--skillname', type=str, default='transfer')
    parser.add_argument('--commit-for', type=int, default=5)
    parser.add_argument('--train-epoch', type=int, default=10)

    parser.add_argument('--batch-size', type=int, default=64)

    
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    run(**args)
    
