import numpy as np 
import math

class Node:
    def __init__(self, state, reward =0., value=0., 
                 skill_num = -1, params = None, parent = None, 
                 height=-1, prob = 0., id_in_memory = None, obj_in_grasp = False):
        self.state = state
        self.child = None
        self.reward = reward
        self.value = value
        self.skill_num = skill_num
        self.params = params
        self.parent = parent
        self.height = height
        self.prob = prob
        self.id_in_memory = id_in_memory
        self.obj_in_grasp = obj_in_grasp

class Planning_with_memories:
    """docstring for Planning_with_memories"""
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
        info = dict()
        
        openlist = []
        leaflist = []

        if height is None:
            # root call
            height = self.max_height
            openlist.append(Node(state=state.copy(), height=height, prob=1., obj_in_grasp = env.obj_grasped(state)))

        while(openlist):
            curr_node = openlist.pop(0)

            # reached a leaf node
            if(curr_node.height==0):
                leaflist.append(curr_node)
                continue

            obj_grasped = curr_node.obj_in_grasp
            if obj_grasped:
                available_skill_set = [1]
            else:
                available_skill_set = [0,2]
            # create child node
            # sample skills
            sampled_skills = np.random.choice(available_skill_set, size = self.num_samples)
            # print(sampled_skills)

            curr_node_obj_loc = curr_node.state[3:6].copy()

            for node_id, sampled_skill_num in enumerate(sampled_skills):
                
                obj_in_grasp = False

                if sampled_skill_num == 0:
                    # sample near obj
                    sampled_params = curr_node_obj_loc + np.random.uniform(low=-0.03,high=0.03, size = curr_node_obj_loc.size)
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

                    sampled_params = curr_node.state[-3:] + np.random.uniform(low=-0.1,high=0.1, size = 3)

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
                    obj_in_grasp = True

                # print("skill:%d, goal"%sampled_skill_num, sampled_params)
                # if (np.any(sampled_params>1) or np.any(sampled_params<-1) ):
                #     print("skill:%d, goal"%sampled_skill_num, sampled_params)
                #     from ipdb import set_trace
                #     set_trace()
                
                critic_value = self.skillset.get_critic_value(primitive_id=sampled_skill_num, obs = curr_node.state, primitive_params = sampled_params)
                next_state, id_in_memory = self.skillset.get_terminal_state_from_memory(primitive_id=sampled_skill_num,obs = curr_node.state, primitive_params = sampled_params)
                prob = self.skillset.get_prob_skill_success(primitive_id=sampled_skill_num,obs = curr_node.state, primitive_params = sampled_params)
                
                # if that edge is not successful then don't use it
                # added leaflist length condition to have atleast 1 path
                if prob<0.5 and len(leaflist)>1:
                    continue

                add_to_openlist = True
                # print("prob:%.4f"%prob)
                if curr_node.height == 1:
                    # creating a leaf node
                    child_reward = self.env.calc_dense_reward(next_state)
                elif (self.env.calc_reward(next_state) == 0):
                    # already reached state
                    child_reward = self.env.calc_dense_reward(next_state)
                    add_to_openlist = False
                else:
                    child_reward = 0.

                new_node = Node(state = next_state, 
                                value= critic_value + curr_node.value, 
                                reward = child_reward + curr_node.reward, 
                                skill_num = sampled_skill_num, params = sampled_params, 
                                parent = curr_node, height = curr_node.height -1,
                                prob = curr_node.prob*prob,
                                id_in_memory = id_in_memory,
                                obj_in_grasp = obj_in_grasp)

                # no need to sort as we want all the paths to be explored
                if add_to_openlist: 
                    openlist.append(new_node)
                else:
                    leaflist.append(new_node)

        max_utility = -math.inf
        max_utility_node_id = -1
        # get the child with max utility
        num_success_plans = 0
        for node_id, node in enumerate(leaflist):
            # print("value:%.4f, reward:%.4f"%(node.value, node.reward))
            # print("skill:%d, memoryid:%d"%(node.skill_num, node.id_in_memory))

            if (node.reward)> -0.03:
                # print("reward:%.4f"%node.reward)
                # print(state[[0,1,2,3,4,5,-3,-2,-1]])
                # print(node.state[[0,1,2,3,4,5,-3,-2,-1]])
                num_success_plans += 1
            # if(node.skill_num>0):
            #   print("obj loc:%s"%(str(node.state[3:6])))
            # else:
            #   print("Reacher")
            node_utility = (self.value_scale*node.value + self.reward_scale*node.reward)

            expected_node_utility = node.prob*node_utility

            if(node_utility > max_utility):
                max_utility = node_utility
                max_utility_node_id = node_id

        # print("num:%d,per success plans:%.4f"%(num_success_plans, num_success_plans/(len(leaflist)+0.)))
        # from ipdb import set_trace
        # set_trace()

        # get the root node and skills
        curr_node = leaflist[node_id]
        while(curr_node.parent is not None):
            parent_node = curr_node.parent
            parent_node.child = curr_node
            curr_node = parent_node

        chosen_skill = curr_node.child.skill_num
        chosen_skill_params = curr_node.child.params

        # create paction and return
        # print("suggested skill id:%d, utility:%.4f,goal:"%(chosen_skill, max_utility), chosen_skill_params)
        paction_orig = self.get_curr_paction(chosen_skill, chosen_skill_params)

        # print("suggested meta action",paction_orig)
        
        info['next_state'] = (curr_node.child.state)
        info['prob'] = curr_node.prob
        # print("prob:%.4f, value:%.4f, goal"%(curr_node.prob, curr_node.child.value), chosen_skill_params)
        
        # create full plan
        info['plan'] = list()
        info['sequence'] = list()
        info['trajectories'] = list()
        first_skill = True
        while(curr_node.child is not None):
            curr_state = curr_node.state
            info['sequence'].append(curr_node.child.skill_num)
            paction = self.get_curr_paction(curr_node.child.skill_num, curr_node.child.params)
            next_state = curr_node.child.state
            reward = self.env.calc_reward(next_state)
            done = (reward==0)
            info['plan'].append((curr_state, paction, reward, next_state, done))

            if first_skill:
                # create trajectory
                curr_traj = self.skillset.get_traj_from_memory(curr_node.child.skill_num, curr_node.child.id_in_memory, curr_node.state)
        
                # first state should be replaced by the real state observed because memory retrieved state is approx
                curr_traj[0] = [curr_state.copy(), curr_traj[0][1],curr_traj[0][2]]
                info['trajectories'].extend(curr_traj)
                first_skill = False

            curr_node = curr_node.child

        return paction_orig.copy(), info




        





        
