from HER.envs import grasping_withgap
import numpy as np 
from gym.envs.robotics.utils import mocap_set_action
from time import sleep 

class BaxterEnv(grasping_withgap.BaxterEnv):
    
    def __init__(self, max_len=50, test = False, filename = "mjc/putaoutb.xml"):
        super(BaxterEnv, self).__init__(max_len=max_len, test=test, filename = filename)

    
    def reset_model(self):
        # randomize gripper loc

        self.close_gripper(gap=0)
        gripper_pos = self.np_random.uniform(self.target_range_min[:self.space_dim] + 0.05, self.target_range_max[:self.space_dim] - 0.05, size=self.space_dim)        
        
        
        target_x, target_y = 0.08*self.np_random.uniform(-1,1, size=2) 
        self.data.set_joint_qpos('target_x', target_x)
        self.data.set_joint_qpos('target_y', target_y)


        # 0: random, 1: grasped
        sample = self.np_random.choice(2)
        #print("sample", sample)
        if sample == 1 and (not self.test):
            # this pose has to be when the obj is inside the container and gripper is holding it

            # move the object out of container so that gripper can open fully
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)
            object_qpos[:self.space_dim] = [0.1,0.1,0.1]
            object_qpos[3:] = np.array([1., 0.,0.,0.])
            self.sim.data.set_joint_qpos('box', object_qpos)

            gripper_pos = np.array([0.6 , 0.3 , 0.1])
            self.close_gripper(gap=0)
            # gripper_pos = self.np_random.uniform(self.target_range_min[:self.space_dim] + 0.05, self.target_range_max[:self.space_dim] - 0.05, size=self.space_dim)        
            self.apply_action(pos=gripper_pos)
            self.sim.step()


            # define one pose in hand 
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)
            object_qpos[:self.space_dim] = gripper_pos[:self.space_dim] - np.array([0., 0., 0.08])

            object_qpos[3:] = np.array([1., 0.,0.,0.])
            self.sim.data.set_joint_qpos('box', object_qpos)

            self.close_gripper(gap=-1)
            self.sim.step()

        else:
            self.apply_action(pos=gripper_pos)
            # spawn near gripper
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)

            dim = 2
            object_qpos[:dim] = self.np_random.uniform([0.52, 0.22], [0.58, 0.28], size=dim) 
            object_qpos[dim] = 0.0

            # print("obj2tar", np.linalg.norm(target_qpos[:dim] - object_qpos[:dim]))
            # spawning obj on ground
            object_qpos[dim] = 0.0

        
        object_qpos[3:] = np.array([1., 0.,0.,0.])
        self.sim.data.set_joint_qpos('box', object_qpos)  

        self.num_step = 1
        return self._get_obs()

    # def calc_reward(self, state, th):
    #     # this functions calculates reward on the current state
    #     gripper_pose = state[:self.space_dim]
    #     obj_pose = state[self.space_dim:2*self.space_dim]
    #     target_pose = state[-self.space_dim:] 
        
    #     ## reward function definition
    #     # print("dist", np.linalg.norm(obj_pose[:2]- target_pose[:2]))
    #     reward_reaching_goal = (np.linalg.norm(obj_pose[:2]- target_pose[:2]) < 0.065) and ((obj_pose[2]- target_pose[2]) < 0.02)
    #     total_reward = -1*(not reward_reaching_goal)
    #     return total_reward

    def calc_dense_reward(self, state):
        # this functions calculates reward on the current state
        gripper_pose = state[:self.space_dim]
        obj_pose = state[self.space_dim:2*self.space_dim]
        target_pose = state[-self.space_dim:] 
        
        ## reward function definition
        return - np.linalg.norm(obj_pose- target_pose)
    
