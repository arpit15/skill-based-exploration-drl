from HER.envs import reacher2d
import numpy as np 

class BaxterEnv(reacher2d.BaxterEnv):
    
    def __init__(self, max_len=50):
        super(BaxterEnv, self).__init__(max_len=max_len, obs_dim=6, action_dim=2, filename="mjc/pusher.xml", space_dim=2)

    
    def reset_model(self, 
        # define one pose in hand 
        gripper_pos = np.array([0.6 , 0.3 , 0.12]),
        add_noise = False,
        randomize_obj = True):

        if randomize_obj:
            # randomize obj loc
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)
            object_qpos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=2)
            # fill obj to fully on ground
            object_qpos[3:] = np.array([1., 0.,0.,0.])
            self.data.set_joint_qpos('box', object_qpos)
            
            target_qpos = self.sim.data.get_joint_qpos('target') 
            target_qpos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=self.space_dim)
                
            while(np.linalg.norm(target_qpos[:self.space_dim] - object_qpos[:self.space_dim]) < 0.05):
                assert target_qpos.shape == (7,)
                target_qpos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=self.space_dim)
                
                self.data.set_joint_qpos('target', target_qpos)
                target_qpos = self.sim.data.get_joint_qpos('target') 

                # print("Setting target qpos to ", target_qpos)
                
            
            gripper_pos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=self.space_dim)
            while(np.linalg.norm(gripper_pos[:self.space_dim] - object_qpos[:self.space_dim]) < 0.1):
                gripper_pos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=self.space_dim)

        return super(BaxterEnv, self).reset_model(gripper_pos = gripper_pos,
            add_noise = add_noise)

    def _get_obs(self):
        ee_pos = self.sim.data.get_site_xpos('grip')[:self.space_dim]
        obj_pos = self.sim.data.get_site_xpos('box')[:self.space_dim]
        target_pos = self.sim.data.get_site_xpos('target')[:self.space_dim]
        
        state = np.concatenate([ee_pos, obj_pos, target_pos])
        return state
        

    def close_gripper(self, gap=0):
        self.data.ctrl[0] = (gap+1)*0.04
        self.data.ctrl[1] = -(gap+1)*0.04
        
    def step(self, action):
        
        ob, total_reward, done, info = super(BaxterEnv, self).step(action)
        
        # check the box location within range
        gripper_pos = ob[:self.space_dim]
        obj_pos = ob[self.space_dim:2*self.space_dim]
        target_pos = ob[-self.space_dim:]

        block_out_of_bound = False
        if obj_pos.size == 2:
            x,y = obj_pos
            if (x<0.3 or x>0.8) or (y<0.0 or y>0.6):
                block_out_of_bound = True
        else:
            x,y,z = obj_pos
            if (x<0.3 or x>0.8) or (y<0.0 or y>0.6) or (z<-0.1 or z>0.25):
                block_out_of_bound = True

        if block_out_of_bound:
            done = True
            info['done'] = 'unstable simulation'
            total_reward -= (self.max_num_steps - self.num_step) + 2

        return ob, total_reward, done, info

    def apply_hindsight(self, states, actions, goal_state):
        '''generates hindsight rollout based on the goal
        '''
        goal = goal_state[self.space_dim:2*self.space_dim]    ## this is the absolute goal location = obj last loc
        # enter the last state in the list
        states.append(goal_state)
        num_tuples = len(actions)

        her_states, her_rewards = [], []
        # make corrections in the first state 
        states[0][-self.space_dim:] = goal.copy()
        her_states.append(states[0])
        for i in range(1, num_tuples + 1):
            state = states[i]
            state[-self.space_dim:] = goal.copy()    # copy the new goal into state
            reward = self.calc_reward(state)
            her_states.append(state)
            her_rewards.append(reward)

        return her_states, her_rewards


    def calc_reward(self, state):
        # this functions calculates reward on the current state
        gripper_pose = state[:self.space_dim]
        obj_pose = state[self.space_dim:2*self.space_dim]
        target_pose = state[-self.space_dim:] 
        
        ## reward function definition
        reward_reaching_goal = np.linalg.norm(obj_pose- target_pose) < 0.05
        total_reward = -1*(not reward_reaching_goal)
        return total_reward

