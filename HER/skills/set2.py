import numpy as np 
from utils import mirror

putAinB = {
	"nb_actions":4,
	"action_func":mirror,
	"skill_name": "putAinB",
	"observation_shape":(9,),
	"obs_func":mirror,
	"num_params":3,
	"restore_path":"/home/arpit/new_RL3/baseline_results_new/clusters/Baxter3dpen-v2/run1/model"
}

pnpBox = {
	"nb_actions":4,
	"action_func":mirror,
	"skill_name": "pnpBox",
	"observation_shape":(9,),
	"obs_func":mirror,
	"num_params":3,
	"restore_path":"/home/arpit/new_RL3/baseline_results_new/HER/Baxter3d-v1/run2/model"
}

skillset = [putAinB, pnpBox]


