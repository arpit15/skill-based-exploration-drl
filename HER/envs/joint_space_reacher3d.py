import numpy as np
import os.path as osp
from gym import utils
from gym.envs.mujoco import mujoco_env

class BaxterEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, filename="mjc/reacher_w_actuators.xml", max_len=50):

        self.num_step = 0   # just a dummy declaration
        self.max_num_steps = max_len
        # limits of motion with some safety limits
        self.target_range_min = np.array([0.3, 0.0, 0.08]) + 0.05
        self.target_range_max = np.array([0.8, 0.6, 0.25]) - 0.05
        
        
        # motion dim
        self.space_dim = 3

        dirname = osp.dirname(osp.abspath(__file__)) 
        mujoco_env.MujocoEnv.__init__(self, osp.join(dirname, filename), 75)
        utils.EzPickle.__init__(self)
        print("ENV INIT DONE!")


    def step(self, a):
        self.num_step += 1
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        total_reward = self.calc_reward(ob)
        
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


        return ob, total_reward, done, info

    def calc_reward(self, state):
        # this functions calculates reward on the current state
        gripper_pose = state[:self.space_dim]
        target_pose = state[-self.space_dim:] 
        
        ## reward function definition
        reward_reaching_goal = np.linalg.norm(gripper_pose- target_pose) < 0.02
        total_reward = -1*(not reward_reaching_goal)
        return total_reward


    def _get_obs(self):
        ee_pos = self.sim.data.get_site_xpos('grip')[:self.space_dim]
        target_pos = self.sim.data.get_site_xpos('target')[:self.space_dim]
        
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            target_pos,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        target_qpos = self.sim.data.get_joint_qpos('target') 
        assert target_qpos.shape == (7,)
        target_qpos[:self.space_dim] = self.np_random.uniform(self.target_range_min[:self.space_dim], self.target_range_max[:self.space_dim], size=self.space_dim)
        # print("Setting target qpos to ", target_qpos)
        self.data.set_joint_qpos('target', target_qpos)

        self.num_step = 0
        return self._get_obs()


    def viewer_setup(self):
        # cam_pos = np.array([0.1, 0.0, 0.7, 0.01, -45., 0.])
        cam_pos = np.array([1.0, 0.0, 0.7, 0.5, -45, 180])
        self.set_cam_position(cam_pos)

    def set_cam_position(self, cam_pos):
        for i in range(3):
            self.sim.model.stat.center[i] = cam_pos[i]
        self.sim.model.stat.extent = cam_pos[3]