# v1.1 params
## this file assumes the following
# action: [delta_x, delta_y, delta_z, gap]
# obs: [gripper_state, block_state, target_xyz]

# simplifying the mapping -1,1 to the corresponding skills ranges 
# like the gripper can't move out of workspace. therefore any neg value is useless

# skills: transfer, transit, grasping with termination condition
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
	x = 0.3 + (x+1)*0.25
	y = 0. + (y+1)*0.3
	z = 0.08 + (z+1)*0.125
	params = np.array((x,y,z))

	final_obs = np.concatenate((obs[:-3] , params))
	return final_obs

def transit_obs(obs,params):
	
	x,y,z = params
	x = 0.3 + (x+1)*0.25
	y = 0. + (y+1)*0.3
	z = 0.08 + (z+1)*0.125 - 0.08
	params = np.array((x,y,z))

	final_obs = np.concatenate((obs[:dim] , params))
	# print("move obs", final_obs)
	return final_obs


def grasp_obs(obs, params):
	# params is the height the obj has to be raised above the ground
	# [-1, 1] -> [0., 0.05]
	obj_loc = obs[dim:2*dim]
	target = [obj_loc[0], obj_loc[1], (params+1)*0.025]
	final_obs = np.concatenate((obs[:-3], target ))
	# print("grasp ob", final_obs)
	return final_obs


def end_transit(obs):
	tmp = obs[dim:2*dim] + np.array([0.,0.,0.1])
	final_obs = np.concatenate((obs[:dim] , tmp))
	return np.linalg.norm(final_obs[:dim] -  final_obs[-dim:]) < 0.05

def end_grasp(obs):
	obj_loc = obs[dim:2*dim]
	target_loc = obs[-dim:]
	return np.linalg.norm(obj_loc-target_loc) < 0.03

end_transfer = end_grasp

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
	"restore_path":"~/new_RL3/baseline_results_new/v1/transfer-v0/run1"
}

skillset = [transit, transfer, grasp]


