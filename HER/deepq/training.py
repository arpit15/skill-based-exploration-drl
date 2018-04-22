import os
import tempfile
from time import sleep

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import gym

from HER import logger
from HER.ddpg.util import normal_mean, normal_std

import baselines.common.tf_util as U
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.simple import ActWrapper

#debug
from ipdb import set_trace

def train(env,
        eval_env,
        q_func,
        lr=5e-4,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=1,
        batch_size=32,
        print_freq=100,
        checkpoint_freq=10000,
        learning_starts=1000,
        gamma=1.0,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        callback=None,
        my_skill_set=None,
        log_dir = None,
        num_eval_episodes=10,
        render=False,
        render_eval = False,
        commit_for = 1
        ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model


    if my_skill_set: assert commit_for>=1, "commit_for >= 1"

    save_idx = 0
    with U.single_threaded_session() as sess:
    

        ## restore
        if my_skill_set:
            action_shape = my_skill_set.len
        else:
            action_shape = env.action_space.n
            
        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph
        observation_space_shape = env.observation_space.shape
        def make_obs_ph(name):
            return U.BatchInput(observation_space_shape, name=name)

        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=action_shape,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            gamma=gamma,
            grad_norm_clipping=10,
            param_noise=param_noise
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': action_shape,
        }

        act = ActWrapper(act, act_params)

        # Create the replay buffer
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = max_timesteps
            beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(buffer_size)
            beta_schedule = None
        # Create the schedule for exploration starting from 1.
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        # sess.run(tf.variables_initializer(new_variables))
        # sess.run(tf.global_variables_initializer())
        update_target()

        if my_skill_set:
            ## restore skills
            my_skill_set.restore_skillset(sess=sess)
            

        episode_rewards = [0.0]
        saved_mean_reward = None
        obs = env.reset()
        reset = True
        
        model_saved = False
        
        model_file = os.path.join(log_dir, "model", "deepq")

        # save the initial act model 
        print("Saving the starting model")
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        act.save(model_file + '.pkl')

        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            paction = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            
            if(my_skill_set):
                skill_obs = obs.copy()
                primitive_id = paction
                rew = 0.
                for _ in range(commit_for):
                
                    ## break actions into primitives and their params    
                    action = my_skill_set.pi(primitive_id=primitive_id, obs = skill_obs.copy(), primitive_params=None)
                    new_obs, skill_rew, done, _ = env.step(action)
                    if render:
                        # print(action)
                        env.render()
                        sleep(0.1)
                    rew += skill_rew
                    skill_obs = new_obs
                    terminate_skill = my_skill_set.termination(new_obs)
                    if done or terminate_skill:
                        break
                    
            else:
                action= paction

                env_action = action
                reset = False
                new_obs, rew, done, _ = env.step(env_action)
                if render:
                    env.render()
                    sleep(0.1)
              


            # Store transition in the replay buffer for the outer env
            replay_buffer.add(obs, paction, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True
                print("Time:%d, episodes:%d"%(t,len(episode_rewards)))

                # add hindsight experience
            

            if t > learning_starts and t % train_freq == 0:
                # print('Training!')
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            # print(len(episode_rewards), episode_rewards[-11:-1])
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
        
            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 50 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    U.save_state(model_file)
                    act.save(model_file + '%d.pkl'%save_idx)
                    save_idx += 1
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
                # else:
                #     print(saved_mean_reward, mean_100ep_reward)

            if (eval_env is not None) and t > learning_starts and t % target_network_update_freq == 0:
                
                # dumping other stats
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("%d time spent exploring", int(100 * exploration.value(t)))

                print("Testing!")
                eval_episode_rewards = []
                eval_episode_successes = []

                for i in range(num_eval_episodes):
                    eval_episode_reward = 0.
                    eval_obs = eval_env.reset()
                    eval_obs_start = eval_obs.copy()
                    eval_done = False
                    while(not eval_done):
                        eval_paction = act(np.array(eval_obs)[None])[0]
                        
                        if(my_skill_set):
                            eval_skill_obs = eval_obs.copy()
                            eval_primitive_id = eval_paction
                            eval_r = 0.
                            for _ in range(commit_for):
                            
                                ## break actions into primitives and their params    
                                eval_action, _ = my_skill_set.pi(primitive_id=eval_primitive_id, obs = eval_skill_obs.copy(), primitive_params=None)
                                eval_new_obs, eval_skill_rew, eval_done, eval_info = eval_env.step(eval_action)
                                # print('env reward:%f'%eval_skill_rew)
                                if render_eval:
                                    print("Render!")
                                    
                                    eval_env.render()
                                    print("rendered!")

                                eval_r += eval_skill_rew
                                eval_skill_obs = eval_new_obs
                                
                                eval_terminate_skill = my_skill_set.termination(eval_new_obs)

                                if eval_done or eval_terminate_skill:
                                    break
                                
                        else:
                            eval_action= eval_paction

                            env_action = eval_action
                            reset = False
                            eval_new_obs, eval_r, eval_done, eval_info = eval_env.step(env_action)
                            if render_eval:
                                # print("Render!")
                                
                                eval_env.render()
                                # print("rendered!")


                        
                        eval_episode_reward += eval_r
                        # print("eval_r:%f, eval_episode_reward:%f"%(eval_r, eval_episode_reward))
                        eval_obs = eval_new_obs
                        
                    eval_episode_success = (eval_info["done"]=="goal reached")
                    if(eval_episode_success):
                        logger.info("success, training epoch:%d,starting config:"%t)


                    eval_episode_rewards.append(eval_episode_reward)
                    eval_episode_successes.append(eval_episode_success)

                combined_stats = {}

                # print(eval_episode_successes, np.mean(eval_episode_successes))
                combined_stats['eval/return'] = normal_mean(eval_episode_rewards)
                combined_stats['eval/success'] = normal_mean(eval_episode_successes)
                combined_stats['eval/episodes'] = (len(eval_episode_rewards))

                for key in sorted(combined_stats.keys()):
                    logger.record_tabular(key, combined_stats[key])
                
                print("dumping the stats!")
                logger.dump_tabular()

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            U.load_state(model_file)
            # act.load(model_file + '.pkl')


