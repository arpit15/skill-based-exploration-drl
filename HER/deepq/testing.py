from baselines import deepq
import os.path as osp

def testing(eval_env, model_path, my_skill_set, render_eval):
    
    act = deepq.load(osp.join(model_path, "model.pkl"))

    ## restore
    if my_skill_set:
        ## restore skills
        my_skill_set.restore_skillset(sess=sess)

    # Evaluate.
    eval_episode_rewards = []
    eval_episode_rewards_history = []
    eval_episode_success = []
    for i in range(10):
        print("Evaluating:%d"%(i+1))
        eval_episode_reward = 0.
        eval_obs = eval_env.reset()
        eval_done = False
        
        while(not eval_done):
            peval_action = act(eval_obs[None])[0]

            if(my_skill_set):
                ## break actions into primitives and their params    
                eval_primitive_id = peval_action
                eval_action, eval_q = my_skill_set.pi(primitive_id=eval_primitive_id, obs = eval_obs.copy(), primitive_params=None)
                
            else:
                eval_action = peval_paction


            eval_obs, eval_r, eval_done, eval_info = eval_env.step(eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            
            if render_eval:
                eval_env.render()
                sleep(0.001)
                
            eval_episode_reward += eval_r

        print("ended",eval_info["done"])
            
        print("episode reward::%f"%eval_episode_reward)
        
        eval_episode_rewards.append(eval_episode_reward)
        eval_episode_rewards_history.append(eval_episode_reward)
        eval_episode_success.append(eval_info["done"]=="goal reached")
        eval_episode_reward = 0.
        
    print("episode reward - mean:%.4f, var:%.4f, success:%.4f"%(np.mean(eval_episode_rewards), np.var(eval_episode_rewards), np.mean(eval_episode_success)))

