from HER.envs import grasping_withgap
import numpy as np 
from gym.envs.robotics.utils import mocap_set_action

class BaxterEnv(grasping_withgap.BaxterEnv):
    
    def __init__(self, max_len=50, test = False, obs_dim=28):
        super(BaxterEnv, self).__init__(max_len=max_len, test=test, obs_dim=obs_dim)

    
    def calc_dense_reward(self, state):
        # this functions calculates reward on the current state
        gripper_pose = state[:self.space_dim]
        obj_pose = state[self.space_dim:2*self.space_dim]
        target_pose = state[-self.space_dim:] 
        
        # print(obj_pose, target_pose)
        ## reward function definition
        return - np.linalg.norm(obj_pose- target_pose)
        
    def reset_model(self):
        # randomize gripper loc

        # 0: random, 1: grasped
        sample = self.np_random.choice(2)
        # print("sample:%d"%sample)
        # randomizing the start state of gripper
        if sample == 0 or self.test:
            gripper_pos = self.np_random.uniform(self.target_range_min[:self.space_dim] + [0.1, 0.1, 0.0], self.target_range_max[:self.space_dim] - [0.1, 0.1, 0.0], size=self.space_dim)
        else:
            gripper_pos = np.array([0.6 , 0.3 , 0.15])

        # print("applied gripper pos", gripper_pos)
        self.apply_action(pos=gripper_pos, ctrl=np.array([0.04, -0.04]))

        if sample == 1 and (not self.test):
            
            # define one pose in hand 
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)
            object_qpos[:self.space_dim] = gripper_pos[:self.space_dim] - np.array([0., 0., 0.1])
            add_noise = False
            self.data.ctrl[:] = np.array([0.,0.])
        else:
            
            # always start with a stable object location. spawning in air is weird

            # print("gripper pos", gripper_pos)
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)

            dim = 2
            object_qpos[:dim] = self.np_random.uniform(self.target_range_min[:dim] + [0.1, 0.1], self.target_range_max[:dim] - [0.1, 0.1], size=dim)
            object_qpos[dim] = 0.

            tmp = 0
            while(np.linalg.norm(gripper_pos[:2] - object_qpos[:2]) < 0.25):
                tmp += 1
                object_qpos[:dim] = self.np_random.uniform(self.target_range_min[:dim] + [0.1, 0.1], self.target_range_max[:dim] - [0.1, 0.1], size=dim)
                object_qpos[dim] = 0.

                if(tmp==100):
                    break
            
            # print("obj pos", object_qpos[:self.space_dim])
            # print("dist b/w obj and gripper:%.4f"%(np.linalg.norm(gripper_pos[:2] - object_qpos[:2])))

        object_qpos[3:] = np.array([1., 0.,0.,0.])
        self.sim.data.set_joint_qpos('box', object_qpos)  

        if sample == 1 and (not self.test):
            # the object drops a bit as the gripper is still adjusting its gap
            self.sim.step()
            object_qpos = self.data.get_joint_qpos('box')

        target_qpos = self.data.get_joint_qpos('target')
        target_qpos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim] + [0.05, 0.05, -0.1], self.target_range_max[:self.space_dim] - [0.05, 0.05, 0.05], size=self.space_dim) 

        # reward threshold is 0.03
        tmp = 0
        while(np.linalg.norm(target_qpos[:self.space_dim] - object_qpos[:self.space_dim]) < 0.25):
            target_qpos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim] + [0.05, 0.05, -0.1], self.target_range_max[:self.space_dim] - [0.05, 0.05, 0.05], size=self.space_dim)
            tmp += 1

            if(tmp==100):
                # print("dist:%.4f, obj:"%(np.linalg.norm(target_qpos[:self.space_dim] - object_qpos[:self.space_dim])), object_qpos[:self.space_dim])
                break
        
        object_qpos = self.sim.data.get_joint_qpos('box') 
        
        self.data.set_joint_qpos('target', target_qpos)
        # print("target pos", target_qpos[:3])
        object_qpos = self.sim.data.get_joint_qpos('box') 
        self.sim.forward()
        self.num_step = 1
        return self._get_obs()

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

