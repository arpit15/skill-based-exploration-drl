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

from HER.pddpg.models import Actor
from HER.successor_prediction_model.v0.models import regressor
import HER.envs
from HER.ddpg.skills import DDPGSkill
import HER.common.tf_util as U

# train_fraction = 0.8

def get_home_path(path):
    curr_home_path = os.getenv("HOME")
    return path.replace("$HOME",curr_home_path)

def generate_data(env, env_id, log_dir, actor, num_ep, commit_for):
    
    log_dir = osp.expanduser(log_dir)
    # get data for training and dump into csv
    csv_filename = osp.join(log_dir, "%s.csv"%env_id)

    # search if the file already exists
    if osp.exists(csv_filename):
        print("Already present!")
        return csv_filename

    # dump the data if the file doesn't exists
    with open(csv_filename, 'w',newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        episode = 0
        while(episode < num_ep):
        # for episode in tqdm(range(num_ep)):
            
            done = False
            starting_ob = env.reset()

            ob = starting_ob
            i = 0
            while(not done or (i<commit_for)):
                action = actor.pi(ob, None)
                ob, _, done, info = env.step(action)
                i += 1

            if(info["done"] != "goal reached"):
                print("didn't succeed")
                continue 

            episode += 1
            #starting_ob = np.concatenate((starting_ob[:6], starting_ob[-3:]))
            writer.writerow(np.concatenate((starting_ob, ob[:-3])).tolist())

    print("DATA logging done!")
    # data input generator
    return csv_filename

##
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/get_started/regression/imports85.py
def in_training_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # If you randomly split the dataset you won't get the same split in both
    # sessions if you stop and restart training later. Also a simple
    # random split won't work with a dataset that's too big to `.cache()` as
    # we are doing here.
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    # Use the hash bucket id as a random number that's deterministic per example
    return bucket_id < int(train_fraction * num_buckets)

def in_test_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # Items not in the training set are in the test set.
    # This line must use `~` instead of `not` because `not` only works on python
    # booleans but we are dealing with symbolic tensors.
    return ~in_training_set(line)
##

def decode_line(line):
    items = tf.decode_csv(line,[[0.]]*(in_size+out_size))
    feats = items[:in_size]
    label = items[in_size:]
    return feats, label

def run(env_id, render, log_dir, restore_dir, commit_for, 
            train_epoch, batch_size=32, lr = 1e-3, seed = 0, dataset_size=2000):
    
    env = gym.make(env_id)
    observation_shape = env.observation_space.shape[-1]
    global in_size, out_size
    in_size = observation_shape
    out_size = observation_shape - 3

    set_global_seeds(seed)
    env.seed(seed)

    
    with U.single_threaded_session() as sess:
        actor_model = DDPGSkill(observation_shape= (observation_shape,), skill_name="skill", nb_actions = env.action_space.shape[-1], restore_path=restore_dir)

        print("Assumption: Goal is 3d target location")
        
        pred_model = regressor(in_shape = in_size, 
                            out_shape = out_size,
                            name = "suc_pred_model", sess=sess,
                            log_dir=log_dir)
        


        init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer(), 
                               # train_iter.initializer, test_iter.initializer
                               )
        sess.run(init_op)

        # restore actor
        actor_model.restore_skill(path = get_home_path(osp.expanduser(restore_dir)), sess = sess)

        generate_data(env, env_id, log_dir, actor_model, dataset_size, commit_for)
        
        exit(1)
        ## creating dataset tensors
        csv_filename = osp.join(log_dir, "%s.csv"%env_id)
        # base_dataset = tf.data.TextLineDataset(csv_filename)

        # train_dataset = base_dataset.filter(in_training_set).map(decode_line).shuffle(buffer_size=5*batch_size, seed =seed).repeat().batch(batch_size)
        # train_iter = train_dataset.make_initializable_iterator()
        # train_el = train_iter.get_next()

        # test_dataset = base_dataset.filter(in_test_set).map(decode_line).batch(batch_size)
        # test_iter = test_dataset.make_initializable_iterator()
        # test_el = test_iter.get_next()

        ## 
        base_dataset = pd.read_csv(csv_filename)
        train, test = train_test_split(base_dataset, test_size=0.2)
        # print(train.shape, test.shape)

        # whiten
        train_mean = np.mean(train, axis=0)
        train_std = np.std(train, axis=0)

        # save mean and var
        statistics = np.concatenate((train_mean, train_std))
        with open(osp.join(log_dir, "%s_stat.npy"%env_id), 'wb') as f:
            np.save(f, statistics)
        # create pd
        train_dataset = ( ( train - train_mean)/train_std)
        test_dataset = ( ( test - train_mean)/train_std)
        test_dataset = test_dataset.values
        test_dataset = [test_dataset[:,:in_size], test_dataset[:,in_size:]]
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
    
