# v0.1 params
## this file assumes the following
# action: [delta_x, delta_y, delta_z, gap]
# obs: [gripper_state, block_state, target]
import numpy as np
from HER.skills.utils import mirror

dim = 3
def move_act(skill_action):
	# get the old gripper loc
	actual_action = [0.]*3 + [-1]
	actual_action[:dim] = skill_action
	return  np.array(actual_action)

def move_obs(obs, params):
	## domain knowledge: move to object
	# obs: gripper, block
	return np.concatenate((obs[:dim], params))

def grasp_obs(obs, params):
	# print("creating grasp obs")
	obj_loc = obs[dim:2*dim]
	target = params.copy()
	final_obs = np.concatenate((obs[:-3], target ))
	# print("grasp ob", final_obs)
	return final_obs


move = {
	"nb_actions":dim,
	"action_func":move_act,
	"skill_name": "move",
	"observation_shape":(dim*2,),
	"obs_func":move_obs,
	"num_params": dim,
	"restore_path":"$HOME/new_RL3/baseline_results_new/v1/Reacher3d-v0/run1/model"
}

grasp = {
	"nb_actions":4,
	"action_func": mirror,
	"skill_name": "grasp",
	"observation_shape":(28,),
	"obs_func":grasp_obs,
	"num_params": 3,
	"restore_path":"$HOME/new_RL3/baseline_results_new/v1/grasping-v2/run2/model"
}

skillset = [move, grasp]


