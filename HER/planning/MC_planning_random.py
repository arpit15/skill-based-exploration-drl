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
        chosen_skill_params = np.random.uniform(low=-1,high=1, size=self.skillset.num_skill_params(chosen_skill))
                 
        # create paction and return
        # print("suggested skill id:%d, utility:%.4f,goal:"%(chosen_skill, max_utility), chosen_skill_params)
        paction_orig = self.get_curr_paction(chosen_skill, chosen_skill_params)

        return paction_orig.copy(), info




        





        
