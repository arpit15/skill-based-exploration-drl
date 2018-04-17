import HER.envs
import sys
import gym
from time import sleep
import numpy as np 
import tensorflow as tf

from HER.ddpg.skills import SkillSet

from ipdb import set_trace
    
if __name__ == "__main__":
    
    print("Loading %s with skillset %s"%(sys.argv[1], sys.argv[2]))
    commit_for = 7
    np.set_printoptions(precision=3)
    eval_env = gym.make(sys.argv[1])
    EVAL_EPISODE = 100
    reward_mat = []

    skillset_file = __import__("HER.skills.%s"%sys.argv[2], fromlist=[''])
    my_skill_set = SkillSet(skillset_file.skillset)

    eval_episode_rewards = []
    eval_episode_rewards_history = []
    eval_episode_success = []

    with tf.Session() as sess:
        # restore skills
        my_skill_set.restore_skillset(sess=sess)

        try:

            for i in range(10):
                print("Evaluating:%d"%(i+1))
                eval_episode_reward = 0.
                eval_obs = eval_env.reset()
                eval_done = False
                
                k = 0
                while(not eval_done):
                    
                    # eval_paction = 1#np.random.choice(my_skill_set.len)
                    if(k<1):
                        eval_paction = 0
                    elif(k<2):
                        eval_paction = 1
                    else:
                        eval_paction = 0
                    
                    if(my_skill_set):
                        eval_skill_obs = eval_obs.copy()
                        eval_primitive_id = eval_paction
                        eval_r = 0.
                        for _ in range(commit_for):
                        
                            ## break actions into primitives and their params    
                            eval_action, _ = my_skill_set.pi(primitive_id=eval_primitive_id, obs = eval_skill_obs.copy(), primitive_params=None)
                            eval_new_obs, eval_skill_rew, eval_done, eval_info = eval_env.step(eval_action)
                            
                            print(eval_paction, eval_action, eval_skill_rew)    
                            eval_env.render()
                            sleep(0.1)
                            
                            eval_r += eval_skill_rew
                            if eval_done:
                                break
                            eval_skill_obs = eval_new_obs
                    
                    eval_episode_reward += eval_r
                    eval_obs = eval_new_obs

                    k += 1
                                
                            

                print("ended",eval_info["done"])
                    
                print("episode reward::%f"%eval_episode_reward)
                
                eval_episode_rewards.append(eval_episode_reward)
                eval_episode_rewards_history.append(eval_episode_reward)
                eval_episode_success.append(eval_info["done"]=="goal reached")
                eval_episode_reward = 0.
                
            print("episode reward - mean:%.4f, var:%.4f, success:%.4f"
                %(np.mean(eval_episode_rewards), 
                    np.var(eval_episode_rewards), 
                    np.mean(eval_episode_success)
                )
                )

        except KeyboardInterrupt:
            print("Exiting!")