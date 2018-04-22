import argparse
import time
import os
import logging

from HER.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from HER.ddpg.skills import SkillSet
from HER.deepq import testing
from HER import logger

import gym
import tensorflow as tf

## my imports
import HER.envs

def run(env_id, seed, evaluation, **kwargs):
    
    # Create envs.
    env = gym.make(env_id)

    # print(env.action_space.shape)
    logger.info("Env info")
    logger.info(env.__doc__)
    logger.info("-"*20)
    gym.logger.setLevel(logging.WARN)
    
    if kwargs['skillset']:
        skillset_file = __import__("HER.skills.%s"%kwargs['skillset'], fromlist=[''])
        my_skill_set = SkillSet(skillset_file.skillset)
    else:
        my_skill_set = None
    
    set_global_seeds(seed)
    env.seed(seed)

    model_path = os.path.join(kwargs['restore_dir'], "model")
    testing.testing(env, model_path, my_skill_set, kwargs['render_eval'], kwargs['commit_for'], kwargs['nb_eval_episodes'])

    env.close()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Baxter3dbox-v0')
    boolean_flag(parser, 'render-eval', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--nb-eval-episodes', type=int, default=100)  # per epoch cycle and MPI worker
    boolean_flag(parser, 'evaluation', default=True)

    ## saving and restoring param parser
    parser.add_argument('--log-dir', type=str, default='/tmp/her')
    parser.add_argument('--restore-dir', type=str, default=None)
    boolean_flag(parser, 'dologging', default=False)
    boolean_flag(parser, 'invert-grad', default=False)
    
    # meta parameters
    parser.add_argument('--commit-for', type=int, default=1)
    parser.add_argument('--skillset', type=str, default='set3')

    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
   
    logger.configure(dir=args["log_dir"])
    logger.info(str(args))
        
    # Run actual script.
    try:
        run(**args)
    except KeyboardInterrupt: 
        print("Exiting!")
