from HER.envs import picknmove_withextras
import numpy as np 
from gym.envs.robotics.utils import mocap_set_action

class BaxterEnv(picknmove_withextras.BaxterEnv):
    
    def __init__(self, max_len=50, test = False):
        super(BaxterEnv, self).__init__(max_len=max_len, test=test)

    def reset_model(self):
        # randomize gripper loc

        gripper_pos = np.array([0.6 , 0.3 , 0.15])

        # print("applied gripper pos", gripper_pos)
        self.apply_action(pos=gripper_pos, ctrl=np.array([0.04, -0.04]))

            
        # define one pose in hand 
        object_qpos = self.sim.data.get_joint_qpos('box') 
        assert object_qpos.shape == (7,)
        object_qpos[:self.space_dim] = gripper_pos[:self.space_dim] - np.array([0., 0., 0.1])
        add_noise = False
        self.data.ctrl[:] = np.array([0.,0.])
        
        # the object drops a bit as the gripper is still adjusting its gap
        self.sim.step()
        object_qpos = self.data.get_joint_qpos('box')

        target_qpos = self.data.get_joint_qpos('target')
        target_qpos[:self.space_dim] = np.array([0.4 , 0.3 , 0.05])

        # reward threshold is 0.03
        
        object_qpos = self.sim.data.get_joint_qpos('box') 
        
        self.data.set_joint_qpos('target', target_qpos)
        # print("target pos", target_qpos[:3])
        object_qpos = self.sim.data.get_joint_qpos('box') 
        self.sim.forward()
        self.num_step = 1
        return self._get_obs()
