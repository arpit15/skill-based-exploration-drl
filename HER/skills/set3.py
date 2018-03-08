## this file assumes the following
# action: [delta_x, delta_y, delta_z, gap]
# obs: [gripper_xyz, block_xyz_wrt_gripper, target_xyz_wrt_block]
import numpy as np
from HER.skills.utils import mirror

dim = 3
def move_act(skill_action):
	actual_action = [0.]*4
	actual_action[:dim] = skill_action
	return  np.array(actual_action)

def move_obs(obs, params):
	## domain knowledge: move to object
	# obs: gripper, block
	return np.concatenate((obs[:dim], params ))

def grasp_obs(obs, params):
	return np.concatenate((obs[:6], params ))


move = {
	"nb_actions":dim,
	"action_func":move_act,
	"skill_name": "move",
	"observation_shape":(dim*2,),
	"obs_func":move_obs,
	"num_params": dim,
	"restore_path":"$HOME/new_RL3/baseline_results_new/HER/Baxter3dReachermod-v1/run2/model"
}

grasp = {
	"nb_actions":4,
	"action_func": mirror,
	"skill_name": "grasp",
	"observation_shape":(9,),
	"obs_func":grasp_obs,
	"num_params": 3,
	"restore_path":"$HOME/new_RL3/baseline_results_new/HER/Baxter3dgraspmod-v1/run3/model"
}

skillset = [move, grasp]


