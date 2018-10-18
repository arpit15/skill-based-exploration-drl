from HER.envs import reacher2d
import numpy as np 
from gym.envs.robotics.utils import mocap_set_action
from gym.envs.robotics import rotations
from time import sleep

class BaxterEnv(reacher2d.BaxterEnv):
    
    def __init__(self, max_len=50, test=False,filename="mjc/gripper.xml", obs_dim=28):
        super(BaxterEnv, self).__init__(max_len=max_len, obs_dim=obs_dim, action_dim=4, filename=filename, space_dim=3)
        self.test = test

    
    def reset_model(self,
                    gripper_pos = np.array([0.6 , 0.3 , 0.15]),
                    ctrl=np.array([0.04, -0.04])
        ):
        
        # 0: random, 1: grasped
        # sample = self.np_random.choice(2)
        sample = 1
        # print("sample:%d"%sample)
        # randomizing the start state of gripper
        if sample == 0 or self.test:
            gripper_pos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim] + [0.1, 0.1, 0.0], self.target_range_max[:self.space_dim] - [0.1, 0.1, 0.0], size=self.space_dim)
        # print("applied gripper pos", gripper_pos)
        self.apply_action(pos=gripper_pos, ctrl=ctrl)


        if (sample == 1) and (not self.test):
            
            # define one pose in hand 
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)
            object_qpos[:self.space_dim] = gripper_pos[:self.space_dim] - np.array([0., 0., 0.1])
            add_noise = False
            self.data.ctrl[:] = np.array([0.,0.])
        else:
            
            # spawn near gripper
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)

            object_qpos[:2] = self.np_random.uniform(self.target_range_min[:2] + [0.1, 0.1], self.target_range_max[:2] - [0.1, 0.1], size=2)
            object_qpos[2] = 0.0

            dim = 2
            while(np.linalg.norm(gripper_pos[:dim] - object_qpos[:dim]) > 0.1
                or
                np.linalg.norm(gripper_pos[:dim] - object_qpos[:dim]) < 0.05):
                object_qpos[:dim] = self.np_random.uniform(self.target_range_min[:dim] + [0.1, 0.1], self.target_range_max[:dim] - [0.1, 0.1], size=dim)

            # print("dist obj and gripper:%.4f"%(np.linalg.norm(gripper_pos[:dim] - object_qpos[:dim])))

        # spawning obj on ground
        if sample == 0: object_qpos[2] = 0.0
        object_qpos[3:] = np.array([1., 0.,0.,0.])
        self.sim.data.set_joint_qpos('box', object_qpos)  

        if sample == 1 and (not self.test):
            # the object drops a bit as the gripper is still adjusting its gap
            self.sim.step()
            object_qpos = self.data.get_joint_qpos('box')

        target_qpos = self.data.get_joint_qpos('target')
        target_qpos[:self.space_dim] = object_qpos[:self.space_dim] + np.array([0., 0., 0.05 + 0.03*(sample==0)*(np.random.rand())])
        
        #print("Setting target qpos to ", target_qpos[:self.space_dim])
        self.data.set_joint_qpos('target', target_qpos)
        self.sim.forward()
        self.num_step = 1

        return self._get_obs()
        


    def _get_obs(self):
        # print("trying to get obs")
        ee_pos = self.sim.data.get_site_xpos('grip')
        obj_pos = self.sim.data.get_site_xpos('box')[:self.space_dim]
        target_pos = self.sim.data.get_site_xpos('target')[:self.space_dim]
        
        # print("targetpos", target_pos, "qpos", self.data.get_joint_qpos('target'))
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # ee vel
        ee_velp = self.sim.data.get_site_xvelp('grip') * dt

        # gripper state
        gripper_state_l = self.data.get_joint_qpos("l_gripper_l_finger_joint")
        gripper_state_r = self.data.get_joint_qpos("l_gripper_r_finger_joint")

        # obj rel pos
        obj_rel_pos = obj_pos - ee_pos
        # obj rotation
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('box'))
        # obj vel
        object_velp = self.sim.data.get_site_xvelp('box') * dt
        object_velr = self.sim.data.get_site_xvelr('box') * dt

        # gripper
        gripper_state = np.array([gripper_state_l, gripper_state_r])
        gripper_vel = gripper_state*dt

        # things missing obj_rel_pos, obj_rot, obj_velr
        state = np.concatenate([ee_pos, obj_pos, obj_rel_pos, gripper_state, object_rot, object_velp, object_velr, ee_velp, gripper_vel, target_pos])
        return state
        

    def close_gripper(self, gap=0):
        self.data.ctrl[0] = (gap+1)*0.04
        self.data.ctrl[1] = -(gap+1)*0.04
        
    def obj_grasped(self, ob):
        # test if obj is grasped. can only be performed on the current state
        # print("checking for grasping")
        # for _ in range(50):
        #     sleep(0.1)
        #     self.render()

        
        assert ob.shape[0]==28, 'obj grasped test if only valid for 28 dim state space'
        gap = ob[9]/0.04 - 1
        self.close_gripper(gap=-1)
        self.do_simulation(n_frames=4)

        new_ob = self._get_obs()
        new_gap = (new_ob[9] - new_ob[10])

        obj_grasped = (new_gap > 1e-2)

        if not obj_grasped:
            # restore the old gap
            self.close_gripper(gap=gap)
            self.do_simulation(n_frames=2)

        # print(new_gap)
        # print("obj grasped ", obj_grasped)
        return obj_grasped



    def step(self, action):
        
        self.num_step += 1
        
        ## parsing of primitive actions
        delta_x, delta_y, delta_z, gripper = action
        
        # cap the motions
        delta_x = max(-1, min(1, delta_x))
        delta_y = max(-1, min(1, delta_y))
        delta_z = max(-1, min(1, delta_z))

        x, y, z = self.get_mocap_state()
        
        new_x = max(0.3, min(0.8, x + delta_x*0.05))
        new_y = max(0.0, min(0.6, y+ delta_y*0.05))
        new_z = max(0.1, min(0.25, z+ delta_z*0.05))

        delta_pos = np.array([new_x - x, new_y - y, new_z - z])
        delta_quat = np.array([0.0, 0.0 , 1.0, 0.])
        delta = np.concatenate((delta_pos, delta_quat))
        mocap_set_action(self.sim, delta)
        self.close_gripper(gripper)
        self.do_simulation()      

        ob = self._get_obs()
        total_reward = self.calc_reward(ob)
        
        box_loc = ob[self.space_dim:2*self.space_dim]

        ## getting state
        info = {"done":None}
        if total_reward == 0:
            done = True
            info["done"] = "goal reached"
        elif (self.num_step > self.max_num_steps):
            done = True
            info["done"] = "max_steps_reached"
        else: 
            done = False

        # check block loc
        obj_pos = ob[self.space_dim:2*self.space_dim]

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
            
        info['absolute_ob'] = ob.copy()
        
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


    def apply_hierarchical_hindsight(self, states, actions, goal_state, sub_states_list):
        '''generates hierarchical hindsight rollout based on the goal
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
            # use all the sub states to calculate total reward
            reward = 0.

            # don't account for the first state : sub_states_list contains first state and last state
            for sub_state in sub_states_list[i-1][1:]:
                sub_state[-self.space_dim:] = goal.copy()    # copy the new goal into state
                reward += self.calc_reward(sub_state)
            her_states.append(state)
            her_rewards.append(reward)

        return her_states, her_rewards
        
    def calc_reward(self, state, threshold=0.03):
        # this functions calculates reward on the current state
        gripper_pose = state[:self.space_dim]
        obj_pose = state[self.space_dim:2*self.space_dim]
        target_pose = state[-self.space_dim:] 
        
        ## reward function definition
        reward_reaching_goal = np.linalg.norm(obj_pose- target_pose) < threshold
        total_reward = -1*(not reward_reaching_goal)
        return total_reward

