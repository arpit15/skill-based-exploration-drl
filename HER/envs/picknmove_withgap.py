from HER.envs import picknmove_withextras
import numpy as np 
from gym.envs.robotics.utils import mocap_set_action

class BaxterEnv(picknmove_withextras.BaxterEnv):
    
    def __init__(self, max_len=50, test = False):
        super(BaxterEnv, self).__init__(max_len=max_len, test=test)

    def _get_obs(self):
        obs = super(BaxterEnv, self)._get_obs()
        grip_pos = obs[:3]
        obj_pos = obs[3:6]
        gripper_state = obs[9:11]

        visible_obs = np.concatenate((grip_pos, obj_pos, gripper_state))

        return visible_obs
    