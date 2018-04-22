# v1
## this file assumes the following
# action: [delta_x, delta_y, delta_z, gap]
# obs: [gripper_state, block_state, target_xyz]

# skills: transfer, transit, grasping without termination condition
import numpy as np
from HER.skills.utils import mirror

dim = 3
def move_act(skill_action, obs):
	# get the old gripper loc
	assert obs.size == 28, "Using with wrong env. The target envs should be picknmove-v3"
	actual_action = [0.]*3 + [-1]#[obs[9]]
	actual_action[:dim] = skill_action
	# print("move action",actual_action)
	return  np.array(actual_action)

def transfer_obs(obs, params):
	## domain knowledge: move to object
	# obs: gripper, target
	# print("creating move obs")
	tmp = obs[-dim:]
	tmp[-1] += 0.1
	final_obs = np.concatenate((obs[:dim] , tmp))
	# print("move obs", final_obs)
	return final_obs

def transit_obs(obs,params):
	tmp = obs[dim:2*dim] + np.array([0.,0.,0.1])
	final_obs = np.concatenate((obs[:dim] , tmp))
	# print("move obs", final_obs)
	return final_obs


def grasp_obs(obs, params):
	# print("creating grasp obs")
	obj_loc = obs[dim:2*dim]
	target = [obj_loc[0], obj_loc[1], 0.05]
	final_obs = np.concatenate((obs[:-3], target ))
	# print("grasp ob", final_obs)
	return final_obs


transit = {
	"nb_actions":dim,
	"action_func":move_act,
	"skill_name": "transit",
	"observation_shape":(dim*2,),
	"obs_func":transit_obs,
	"num_params": dim,
	"restore_path":"$HOME/new_RL3/baseline_results_new/v1/Reacher3d-v0/run1/model"
}

transfer = {
	"nb_actions":dim,
	"action_func":move_act,
	"skill_name": "transfer",
	"observation_shape":(dim*2,),
	"obs_func":transfer_obs,
	"num_params": dim,
	"restore_path":"$HOME/new_RL3/baseline_results_new/v1/Reacher3d-v0/run1/model"
}

grasp = {
	"nb_actions":4,
	"action_func": mirror,
	"skill_name": "grasp",
	"observation_shape":(28,),
	"obs_func":grasp_obs,
	"num_params": dim,
	"restore_path":"$HOME/new_RL3/baseline_results_new/v1/grasping-v2/run2/model"
}

skillset = [transit, transfer, grasp]


