import argparse
import time
import os
import logging

from HER.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from HER.ddpg.skills import SkillSet
from HER.deepq import (training, models)
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

    if evaluation:
        if kwargs['eval_env_id']: 
            eval_env_id = kwargs['eval_env_id']
        else: 
            eval_env_id = env_id
        eval_env = gym.make(eval_env_id)
        # del eval_env_id from kwargs
        del kwargs['eval_env_id']
    else:
        eval_env = None

    
    if kwargs['skillset']:
        skillset_file = __import__("HER.skills.%s"%kwargs['skillset'], fromlist=[''])
        my_skill_set = SkillSet(skillset_file.skillset)

    model = models.mlp([64])

    # Seed everything to make things reproducible.
    logger.info('seed={}, logdir={}'.format(seed, logger.get_dir()))
    
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    start_time = time.time()
    
    training.train(
        env=env,
        eval_env = eval_env,
        q_func=model,
        lr=kwargs['lr'],
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.002,
        train_freq=1,
        batch_size=kwargs['batch_size'],
        print_freq=100,
        checkpoint_freq=50,
        learning_starts=max(50, kwargs['batch_size']),
        target_network_update_freq=100,
        prioritized_replay= kwargs['prioritized_replay'],
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        gamma = kwargs['gamma'],
        log_dir = kwargs['log_dir'],
        my_skill_set= my_skill_set,
        num_eval_episodes=kwargs['num_eval_episodes'],
        render = kwargs['render'],
        render_eval = kwargs['render_eval'],
        commit_for = kwargs['commit_for']
    )
    
    env.close()
    if eval_env is not None:
        eval_env.close()
    
    logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Baxter3dbox-v0')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'render', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--nb-epochs', type=int, default=200)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=40)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=320)  # per epoch cycle and MPI worker
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=True)
    parser.add_argument('--eval-env-id', type=str, default=None)
    parser.add_argument('--num-eval-episodes', type=int, default=10)
    boolean_flag(parser, 'prioritized_replay', default=True)

    ## saving and restoring param parser
    parser.add_argument('--log-dir', type=str, default='/tmp/her')
    parser.add_argument('--save-freq', type=int, default=1)
    parser.add_argument('--restore-dir', type=str, default=None)
    parser.add_argument('--skillset', type=str, default='set4')

    # meta parameters
    parser.add_argument('--commit-for', type=int, default=1)

    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    from ipdb import set_trace
    args = parse_args()
   
    logger.configure(dir=args["log_dir"])
    logger.info(str(args))
    
    # print(logger.Logger.CURRENT.output_formats)
    # set_trace()
    # Run actual script.
    try:
        run(**args)
    except KeyboardInterrupt: 
        print("Exiting!")
