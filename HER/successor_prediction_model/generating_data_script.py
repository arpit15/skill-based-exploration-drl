import argparse
import os
import gym
import csv
import os.path as osp
import tensorflow as tf
import numpy as np 
from tqdm import tqdm

import HER.envs
from HER.successor_prediction_model.models import classifier
from HER.ddpg.skills import DDPGSkill
import HER.common.tf_util as U
from HER.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

def get_home_path(path):
    curr_home_path = os.getenv("HOME")
    return path.replace("$HOME",curr_home_path)


def generate_data(env, env_id, log_dir, actor, num_ep, commit_for):
    
    # get data for training and dump into csv
    csv_filename = osp.join(log_dir, "%s.csv"%env_id)

    # search if the file already exists
    if osp.exists(csv_filename):
        print("Already present!")
        return csv_filename

    # dump the data if the file doesn't exists
    with open(csv_filename, 'w',newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for episode in tqdm(range(num_ep)):
            
            done = False
            starting_ob = env.reset()

            ob = starting_ob
            i = 0
            while(not done or (i<commit_for)):
                action = actor.pi(ob, None)
                ob, _, done, _ = env.step(action)
                i += 1

            writer.writerow(starting_ob.tolist().append(done*1))

    print("DATA logging done!")
    # data input generator
    return csv_filename

def run():
  with U.single_threaded_session() as sess:
    actor_model = DDPGSkill(observation_shape= (observation_shape,), skill_name="skill", nb_actions = env.action_space.shape[-1])

    print("Assumption: Goal is 3d target location")

    pred_model = classifier(in_shape = in_size, 
                        out_shape = out_size,
                        name = "suc_pred_model", sess=sess,
                        log_dir=log_dir)



    init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer(), 
                           )
    sess.run(init_op)

    # restore actor
    actor_model.restore_skill(path = get_home_path(restore_dir), sess = sess)

    generate_data(env, env_id, log_dir, actor_model, dataset_size, commit_for)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Baxter-v1')
    boolean_flag(parser, 'render', default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--log-dir', type=str, default='/tmp/her')
    parser.add_argument('--restore-dir', type=str, default=None)

    parser.add_argument('--dataset-size', type=int, default=2000)
    parser.add_argument('--commit-for', type=int, default=5)

    
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    run(**args)
