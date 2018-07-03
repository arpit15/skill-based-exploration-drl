import numpy as np 
import math

class Planning_with_memories:
    """look ahead with successor model prediction and biased sampling of parameters"""

    def __init__(self, skillset, env, num_samples=2, max_height =3, value_scale=0., reward_scale=1.):
        self.skillset = skillset
        self.env = env
        self.num_samples = num_samples
        self.max_height = max_height

        # value_scale*(sum of values) + reward_scale*(sum of rewards)
        self.value_scale = value_scale
        self.reward_scale = reward_scale
        
    def get_curr_paction(self, skill_num, params):
        paction = np.zeros((self.skillset.len + self.skillset.num_params, ))
        paction[skill_num] = 1.0
        starting_idx = self.skillset.params_start_idx[skill_num]
        ending_idx = starting_idx + params.size

        paction[ self.skillset.len + starting_idx: self.skillset.len + ending_idx] = params
        return paction
        

    def create_plan(self, env, state, height = None):
        
        # print("CREATING PLAN")
        # np.set_printoptions(precision=3)
        

        info = dict()
        available_skill_set = list(range(self.skillset.len))
        sampled_skills = np.random.choice(available_skill_set, size = self.num_samples)
        chosen_skill = sampled_skills[0]
        sampled_skill_num = chosen_skill

        if sampled_skill_num == 0:
            # sample near obj
            sampled_params = state[3:6] + np.random.uniform(low=-0.03,high=0.03, size = 3)
            # convert to skill coordinates
            sampled_params[0] = (sampled_params[0]- 0.3)/0.25 - 1
            sampled_params[1] = (sampled_params[1] - 0.)/0.3 - 1
            sampled_params[2] = (sampled_params[2] + 0.13 - 0.13)/0.035 -1

            # make them between -1,1
            sampled_params[0] = max(min(1, sampled_params[0]), -1)
            sampled_params[1] = max(min(1, sampled_params[1]), -1)
            sampled_params[2] = max(min(1, sampled_params[2]), -1)

        elif sampled_skill_num == 1:
            # sample near target
            # print(state[-3:], curr_node.state[-3:])

            sampled_params = state[-3:] + np.random.uniform(low=-0.1,high=0.1, size = 3)

            # convert to skill coordinates
            sampled_params[0] = (sampled_params[0]- 0.45)/0.1 - 1
            sampled_params[1] = (sampled_params[1] - 0.15)/0.15 - 1
            sampled_params[2] = (sampled_params[2] - 0.03)/0.035 -1

            # make them between -1,1
            sampled_params[0] = max(min(1, sampled_params[0]), -1)
            sampled_params[1] = max(min(1, sampled_params[1]), -1)
            sampled_params[2] = max(min(1, sampled_params[2]), -1)
            
        else:
            # sample any height for grasping
            sampled_params = np.random.uniform(low=-1,high=1, size=self.skillset.num_skill_params(sampled_skill_num))
            ## assume in hallucination that after this skill is performed we will have obj in grasp
            
        chosen_skill_params = sampled_params

        # create paction and return
        # print("suggested skill id:%d, utility:%.4f,goal:"%(chosen_skill, max_utility), chosen_skill_params)
        paction_orig = self.get_curr_paction(chosen_skill, chosen_skill_params)

        return paction_orig.copy(), info




        





        
