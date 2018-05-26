# v2.3 params
## this file assumes the following
# action: [delta_x, delta_y, delta_z, gap]
# obs: [gripper_state, block_state, target_xyz]

# simplifying the mapping -1,1 to the corresponding skills ranges 
# like the gripper can't move out of workspace. therefore any neg value is useless

# skills: transfer, transit, grasping with termination condition
# dynamics model is conditioned on skills

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
	# print("creating transfer obs ", obs.shape, params.shape)

	x,y,z = params
	x = 0.45 + (x+1)*0.1
	y = 0.15 + (y+1)*0.15
	z = 0.03 + (z+1)*0.035
	params = np.array((x,y,z))
	# print("transfer params",params)
	final_obs = np.concatenate((obs[:-3] , params))
	return final_obs

def transit_obs(obs,params):
	# print(params)
	x,y,z = params
	x = 0.3 + (x+1)*0.25
	y = 0. + (y+1)*0.3
	z = 0.13 + (z+1)*0.035
	params = np.array((x,y,z))
	# print("transit params",params)
	
	final_obs = np.concatenate((obs[:dim] , params))
	# print("move obs", final_obs)
	return final_obs


def grasp_obs(obs, params):
	# params is the height the obj has to be raised above the ground
	# [-1, 1] -> [0.05, 0.08]
	obj_loc = obs[dim:2*dim]
	target = [obj_loc[0], obj_loc[1], 0.05+(float(params[0])+1)*0.015]
	final_obs = np.concatenate((obs[:-3], target ))
	# print("grasp ob", final_obs)
	return final_obs


def end_transit(obs, params):
	skill_obs = transit_obs(obs,params)
	# tmp = skill_obs[dim:2*dim] + np.array([0.,0.,0.1])
	# final_obs = np.concatenate((skill_obs[:dim] , tmp))

	# print("transit end",skill_obs[:dim], skill_obs[-dim:])
	return np.linalg.norm(skill_obs[:dim] -  skill_obs[-dim:]) < 0.05

def end_grasp(obs, params):
	skill_obs = grasp_obs(obs, params)

	obj_loc = skill_obs[dim:2*dim]
	target_loc = skill_obs[-dim:]
	return np.linalg.norm(obj_loc-target_loc) < 0.03

def end_transfer(obs, params):
	skill_obs = transfer_obs(obs, params)

	obj_loc = skill_obs[dim:2*dim]
	target_loc = skill_obs[-dim:]
	return np.linalg.norm(obj_loc-target_loc) < 0.03

def reacher_get_full_state_func(reacher_obs, prev_obs):
	# assumption: the gripper vel and everything else is 0 for reacher
	curr_full_obs = np.zeros_like(prev_obs)
	curr_full_obs[:dim] = reacher_obs[:dim].copy()
	curr_full_obs[dim:2*dim] = prev_obs[dim:2*dim].copy()
	curr_full_obs[2*dim: 3*dim] = prev_obs[dim:2*dim] - curr_full_obs[:dim]

	curr_full_obs[-dim:] = reacher_obs[-dim:].copy()
	return curr_full_obs.copy()


transit = {
	"nb_actions":dim,
	"action_func":move_act,
	"skill_name": "transit",
	"observation_shape":(dim*2,),
	"obs_func":transit_obs,
	"num_params": dim,
	"termination": end_transit,
	'get_full_state_func': reacher_get_full_state_func,
	"restore_path":"~/new_RL3/baseline_results_new/v1/Reacher3d-v0/run1"
}

grasp = {
	"nb_actions":4,
	"action_func": mirror,
	"skill_name": "grasp",
	"observation_shape":(28,),
	"obs_func":grasp_obs,
	"num_params": 1,
	"termination": end_grasp,
	"next_state_query_idx":[3,4,5],
	"restore_path":"~/new_RL3/baseline_results_new/v1/grasping-v2/run2"
}

transfer = {
	"nb_actions":4,
	"action_func": mirror,
	"skill_name": "transfer",
	"observation_shape":(28,),
	"obs_func":transfer_obs,
	"num_params": 3,
	"termination": end_transfer,
	"next_state_query_idx":[0,1,2, 3,4,5,25,26,27],
	"restore_path":"~/new_RL3/baseline_results_new/clusters-v1/transfer-v0/run2"
}

skillset = [transit, transfer, grasp]


