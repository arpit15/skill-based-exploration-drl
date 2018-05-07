from HER.envs import picknmove_withextras
import numpy as np 
from gym.envs.robotics.utils import mocap_set_action

class BaxterEnv(picknmove_withextras.BaxterEnv):
    
    def __init__(self, max_len=50, test = False):
        super(BaxterEnv, self).__init__(max_len=max_len, test=test)

    def calc_reward(self, state):
    	# this functions calculates reward on the current state
        gripper_pose = state[:self.space_dim]
        obj_pose = state[self.space_dim:2*self.space_dim]
        target_pose = state[-self.space_dim:] 
        
        ## reward function definition
        dist = np.linalg.norm(obj_pose- target_pose)
        reward_reaching_goal =  (dist< 0.05)
        total_reward = -1*(not reward_reaching_goal) - dist
        return total_reward

    