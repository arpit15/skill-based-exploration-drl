from gym.envs.robotics import FetchPickAndPlaceEnv
import numpy as np

class FetchPnp(FetchPickAndPlaceEnv):
	def _get_obs(self):
		parent_dict = super(FetchPnp, self)._get_obs()
		obs = parent_dict['observation']
		grip_pos = obs[:3]
		obj_pos = obs[3:6]
		obj_rel_pos = obs[6:9]
		gripper_state = obs[9:11]
		obj_rot = obs[11:14]
		obj_velp = obs[14:17]
		obj_velr = obs[17:21]
		grip_vel = obs[21:24]
		gripper_vel = obs[24:26]
		parent_dict['observation'] = np.concatenate((grip_pos, obj_pos, obj_rel_pos, gripper_state, obj_velp, grip_vel, gripper_vel))

		return parent_dict
