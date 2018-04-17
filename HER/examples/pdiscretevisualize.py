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
    

    model_path = os.path.join(kwargs['restore_dir'], "model")
    testing.testing(env, model_path, my_skill_set, kwargs['render_eval'], kwargs['commit_for'])

    env.close()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Baxter3dbox-v0')
    boolean_flag(parser, 'render-eval', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)  # per MPI worker
    parser.add_argument('--nb-epochs', type=int, default=200)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=40)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=320)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='epsnorm_0.01_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=True)

    ## saving and restoring param parser
    parser.add_argument('--log-dir', type=str, default='/tmp/her')
    parser.add_argument('--save-freq', type=int, default=1)
    parser.add_argument('--restore-dir', type=str, default=None)
    boolean_flag(parser, 'dologging', default=True)
    boolean_flag(parser, 'invert-grad', default=False)
    boolean_flag(parser, 'her', default=True)
    boolean_flag(parser, 'actor-reg', default=True)
    boolean_flag(parser, 'tf-sum-logging', default=False)

    # meta parameters
    parser.add_argument('--commit-for', type=int, default=1)

    parser.add_argument('--skillset', type=str, default='set3')

    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
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
