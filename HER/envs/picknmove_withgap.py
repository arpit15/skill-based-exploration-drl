from HER.envs import picknmove_withextras
import numpy as np 
from gym.envs.robotics.utils import mocap_set_action

class BaxterEnv(picknmove_withextras.BaxterEnv):
    
    def __init__(self, max_len=50, test = False,obs_dim=11):
        super(BaxterEnv, self).__init__(max_len=max_len, test=test, obs_dim=obs_dim)

    def _get_obs(self):
        obs = super(BaxterEnv, self)._get_obs()
        grip_pos = obs[:3]
        obj_pos = obs[3:6]
        obj_rel_pos = obs[6:9]
        gripper_state = obs[9:11]
        gripper_vel = obs[24:26]
        target_pos = obs[-3:]


        visible_obs = np.concatenate((grip_pos, obj_pos, obj_rel_pos, gripper_state, gripper_vel, target_pos))

        return visible_obs
    