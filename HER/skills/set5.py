## this file assumes the following
# action: [delta_x, delta_y, delta_z, gap]
# obs: [gripper_xyz, block_xyz_wrt_gripper, target_xyz_wrt_block]
import numpy as np
from HER.skills.utils import mirror

dim = 3
def move_act(skill_action, obs):
	# get the old gripper loc
	assert obs.size == 28, "Using with wrong env. The target envs should be picknmove-v3"
	actual_action = [0.]*3 + [-1]#[obs[9]]
	actual_action[:dim] = skill_action
	return  np.array(actual_action)

def move_obs(obs, params):
	## domain knowledge: move to object
	# obs: gripper, block/target
	obj_rel_pos = obs[6:9]
	if(np.linalg.norm(obj_rel_pos[:2]) < 0.05 and obj_rel_pos[2]<0.15):
		# object in hand
		# print("obj in hand")
		tmp = obs[-dim:]
		tmp[-1] += 0.1
		
	else:
		tmp = obs[dim:2*dim]

	return np.concatenate((obs[:dim] , tmp))

def grasp_obs(obs, params):
	obj_loc = obs[dim:2*dim]
	target = [obj_loc[0], obj_loc[1], 0.05]
	return np.concatenate((obs[:-3], target ))


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


