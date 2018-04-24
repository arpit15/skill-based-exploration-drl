import argparse
import os
import gym
import csv
import os.path as osp
import tensorflow as tf

from HER.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from HER.pddpg.models import Actor
from HER.successor_prediction_model.models import regressor
import HER.envs
from HER.ddpg.skills import DDPGSkill
import HER.common.tf_util as U

def get_home_path(path):
    curr_home_path = os.getenv("HOME")
    return path.replace("$HOME",curr_home_path)

def generate_data(env, env_id, log_dir, actor, sess):
    
    # get data for training and dump into csv
    csv_filename = osp.join(log_dir, "%s.csv"%env_id)

    # search if the file already exists
    if osp.exists(csv_filename):
        return csv_filename

    # dump the data if the file doesn't exists
    with open(csv_filename, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for episode in range(num_ep):
            done = False
            starting_ob = env.reset()

            ob = starting_ob
            while(not done):
                action = actor.pi(ob)
                ob, _, done, _ = env.step(action)

            writer.writerow(np.concatenate((starting_ob[-3:], ob)).tolist())

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
    items = tf.decode_csv(line,[0.]*(in_size+out_size))
    feats = items[:in_size]
    label = items[in_size:]
    return feats, labels

def run(env_id, render, num_ep, log_dir, restore_dir, commit_for, 
            train_epoch, batch_size=32, lr = 1e-3, seed = 0):
    
    env = gym.make(env_id)
    observation_shape = env.observation_space.shape[-1]
    global in_size, out_size
    in_size = observation_shape
    out_size = observation_shape - 3

    set_global_seeds(seed)
    env.seed(seed)

    
    with U.single_threaded_session() as sess:
        actor_model = DDPGSkill(observation_shape= (observation_shape,), skill_name="skill", nb_actions = env.action_space.shape[-1])

        print("Assumption: Goal is 3d target location")
        
        pred_model = regressor(in_shape = observation_shape, 
                            out_shape = observation_shape - 3,
                            name = "suc_pred_model", sess=sess)
        
        ## creating dataset tensors
        csv_filename = osp.join(log_dir, "%s.csv"%env_id)
        base_dataset = tf.data.TextLineDataset(csv_filename)

        train_dataset = base_dataset.filter(in_training_set).map(decode_line).shuffle(buffer_size=5*batch_size, seed =seed).repeat()
        test_dataset = base_dataset.filter(in_test_set).map(decode_line)
        ####

        init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
        sess.run(init_op)

        # restore actor
        actor_model.restore_skill(path = get_home_path(restore_dir), sess = sess)

        generate_data(env, env_id, log_dir, actor_model)

        pred_model.train(sess, num_ep, batch_size, lr, train_dataset, test_dataset)
        pred_model.save()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Baxter-v1')
    boolean_flag(parser, 'render', default=False)
    parser.add_argument('--num-ep', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--log-dir', type=str, default='/tmp/her')
    parser.add_argument('--restore-dir', type=str, default=None)

    # parser.add_argument('--skillset', type=str, default='set8')
    # parser.add_argument('--skillname', type=str, default='transfer')
    parser.add_argument('--commit-for', type=int, default=1)
    parser.add_argument('--train-epoch', type=int, default=10)

    
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    run(**args)
    