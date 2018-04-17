import os.path as osp
from time import sleep
import cloudpickle
import zipfile
import tempfile
import tensorflow as tf
import numpy as np 

from baselines import deepq
import baselines.common.tf_util as U
from baselines.deepq.simple import ActWrapper

def load_actor(sess, model_path):
    model_file = osp.join(model_path,'deepq.pkl')
    print("Loading act model from %s"%(model_file))
    with open(model_file, "rb") as f:
        model_data, act_params = cloudpickle.load(f)
    act = deepq.build_act(**act_params)

    with tempfile.TemporaryDirectory() as td:
        arc_path = osp.join(td, "packed.zip")
        with open(arc_path, "wb") as f:
            f.write(model_data)

        zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            
        fname = osp.join(td, "model")
        saver = tf.train.Saver()
        saver.restore(sess, fname)
    return ActWrapper(act, act_params)
    
def testing(eval_env, model_path, my_skill_set, render_eval, commit_for):
    
    with U.single_threaded_session() as sess:

        act = load_actor(sess, model_path)

        ## restore
        if my_skill_set:
            # restore skills
            my_skill_set.restore_skillset(sess=sess)

        
        # act = deepq.load(osp.join(model_path, "deepq.pkl"))
        # U.load_state( osp.join(model_path , "deepq"))

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
                eval_paction = act(np.array(eval_obs)[None])[0]
                
                if(my_skill_set):
                    eval_skill_obs = eval_obs.copy()
                    eval_primitive_id = eval_paction
                    eval_r = 0.
                    for _ in range(commit_for):
                    
                        ## break actions into primitives and their params    
                        eval_action, _ = my_skill_set.pi(primitive_id=eval_primitive_id, obs = eval_skill_obs.copy(), primitive_params=None)
                        eval_new_obs, eval_skill_rew, eval_done, eval_info = eval_env.step(eval_action)
                        if render_eval:
                            # print("Render!")
                            
                            eval_env.render()
                            sleep(0.1)
                            # print("rendered!")

                        eval_r += eval_skill_rew
                        if eval_done:
                            break
                        eval_skill_obs = eval_new_obs
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
                eval_obs = eval_new_obs
                            
                        

            print("ended",eval_info["done"])
                
            print("episode reward::%f"%eval_episode_reward)
            
            eval_episode_rewards.append(eval_episode_reward)
            eval_episode_rewards_history.append(eval_episode_reward)
            eval_episode_success.append(eval_info["done"]=="goal reached")
            eval_episode_reward = 0.
            
        print("episode reward - mean:%.4f, var:%.4f, success:%.4f"%(np.mean(eval_episode_rewards), np.var(eval_episode_rewards), np.mean(eval_episode_success)))

