from HER.envs import pusher
import numpy as np 

class BaxterEnv(pusher.BaxterEnv):
    
    def __init__(self, max_len=50):
        super(BaxterEnv, self).__init__(max_len=max_len)

    def reset_model(self,
        # define one pose in hand 
        gripper_pos = np.array([0.6 , 0.3 , 0.12]),
        add_noise = False):

        # randomize obj loc
        object_qpos = self.sim.data.get_joint_qpos('box') 
        assert object_qpos.shape == (7,)
        object_qpos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=2)
        # fill obj to fully on ground
        object_qpos[3:] = np.array([1., 0.,0.,0.])
        self.data.set_joint_qpos('box', object_qpos)
        
        target_qpos = self.sim.data.get_joint_qpos('target') 
        target_qpos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=self.space_dim)
            
        while(
            np.linalg.norm(target_qpos[:self.space_dim] - object_qpos[:self.space_dim]) < 0.05
            or
            np.linalg.norm(target_qpos[:self.space_dim] - object_qpos[:self.space_dim]) > 0.1
            ):
            assert target_qpos.shape == (7,)
            target_qpos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=self.space_dim)
            
            self.data.set_joint_qpos('target', target_qpos)
            target_qpos = self.sim.data.get_joint_qpos('target') 

            # print("Setting target qpos to ", target_qpos)

        #print("obj2target",np.linalg.norm(target_qpos[:self.space_dim] - object_qpos[:self.space_dim]))
            
        
        gripper_pos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=self.space_dim)
        while(
            np.linalg.norm(gripper_pos[:self.space_dim] - object_qpos[:self.space_dim]) < 0.07
            or
            np.linalg.norm(gripper_pos[:self.space_dim] - object_qpos[:self.space_dim]) > 0.2
            ):
            gripper_pos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=self.space_dim)

        #print("obj2gripper",np.linalg.norm(gripper_pos[:self.space_dim] - object_qpos[:self.space_dim]))
        
        # print("obj_pos:",object_qpos[:self.space_dim])
        return super(BaxterEnv, self).reset_model(gripper_pos = gripper_pos,
                                            add_noise = add_noise,
                                            randomize_obj = False)

    
