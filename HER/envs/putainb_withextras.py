from HER.envs import grasping_withgap
import numpy as np 
from gym.envs.robotics.utils import mocap_set_action

class BaxterEnv(grasping_withgap.BaxterEnv):
    
    def __init__(self, max_len=50, test = False, filename = "mjc/putainb.xml"):
        super(BaxterEnv, self).__init__(max_len=max_len, test=test, filename = filename)

    
    def reset_model(self):
        # randomize gripper loc

        self.close_gripper(gap=0)
        gripper_pos = self.np_random.uniform(self.target_range_min[:self.space_dim] + 0.05, self.target_range_max[:self.space_dim] - 0.05, size=self.space_dim)        
        
        target_qpos = self.data.get_joint_qpos('target')
        target_qpos[:self.space_dim] = np.array([0.5, 0.3,0.])
        self.data.set_joint_qpos('target', target_qpos)


        # 0: random, 1: grasped
        sample = 0#self.np_random.choice(2)
        print("sample", sample)
        if sample == 1 and (not self.test):

            gripper_pos = np.array([0.5 , 0.32 , 0.2])
            self.close_gripper(gap=0)
            # gripper_pos = self.np_random.uniform(self.target_range_min[:self.space_dim] + 0.05, self.target_range_max[:self.space_dim] - 0.05, size=self.space_dim)        
            self.apply_action(pos=gripper_pos)

            # define one pose in hand 
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)
            object_qpos[:self.space_dim] = gripper_pos[:self.space_dim] - np.array([0., 0., 0.08])

            self.close_gripper(gap=-1)
            self.sim.step()

        else:
            self.apply_action(pos=gripper_pos)
            # spawn near gripper
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)

            dim = 2
            object_qpos[:dim] = self.np_random.uniform(self.target_range_min[:dim] + 0.05, self.target_range_max[:dim] - 0.05 , size=dim) 
            object_qpos[dim] = 0.0

            
            while((np.linalg.norm(gripper_pos[:dim] - object_qpos[:dim]) < 0.05) or
                (np.linalg.norm(target_qpos[:dim] - object_qpos[:dim]) < 0.2)):
                object_qpos[:dim] = self.np_random.uniform(self.target_range_min[:dim] + 0.05, self.target_range_max[:dim] - 0.05, size=dim)

            # print("obj2tar", np.linalg.norm(target_qpos[:dim] - object_qpos[:dim]))
            # spawning obj on ground
            object_qpos[dim] = 0.0

        
        object_qpos[3:] = np.array([1., 0.,0.,0.])
        self.sim.data.set_joint_qpos('box', object_qpos)  

        self.num_step = 1
        return self._get_obs()

    def calc_reward(self, state):
        # this functions calculates reward on the current state
        gripper_pose = state[:self.space_dim]
        obj_pose = state[self.space_dim:2*self.space_dim]
        target_pose = state[-self.space_dim:] 
        
        ## reward function definition
        # print("dist", np.linalg.norm(obj_pose[:2]- target_pose[:2]))
        reward_reaching_goal = (np.linalg.norm(obj_pose[:2]- target_pose[:2]) < 0.065) and ((obj_pose[2]- target_pose[2]) < 0.02)
        total_reward = -1*(not reward_reaching_goal)
        return total_reward
    