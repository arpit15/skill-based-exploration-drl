import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from gym import spaces
import os 
import os.path as osp
import signal 
from math import pi, atan2

import mujoco_py

from trac_ik_python.trac_ik_wrap import TRAC_IK
from trac_ik_python import trac_ik_wrap as tracik

from time import sleep

from ipdb import set_trace

class BaxterEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """cts env, 6dim
    state space: relative state space position of gripper, (block-gripper) and (target-block)
    random restarts for block and target on the table
    reward function: - 1(not reaching)
    actions: (delta_x, delta_y) 5cm push
    starting state: (0.63, 0.2, 0.59, 0.27, 0.55, 0.3)
    max_num_steps = 50
    """
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__)) 
        mujoco_env.MujocoEnv.__init__(self, os.path.join(dirname, "mjc/baxter_orient_left_cts.xml") , 1)
        utils.EzPickle.__init__(self)

        ## mujoco things
        # task space action space

        low = np.array([-1., -1.])
        high = np.array([1., 1.])

        self.action_space = spaces.Box(low, high)

        self.tuck_pose = {
                            'left':  np.array([-0.08, -1.0, -1.19, 1.94,  0.67, 1.03, -0.50])
                       }

        self.start_pose = {
                            'left' : np.array([-0.21, -0.75, -1.4, 1.61, 0.60, 0.81, -0.52])
                            }
        

        ## starting pose
        self.init_qpos = self.data.qpos.copy().flatten()
        self.init_qpos[1:8] = np.array(self.start_pose["left"]).T
        
        
        ## ik setup
        urdf_filename = osp.join(dirname, "urdf", "baxter.urdf")
                
        with open(urdf_filename) as f:
            urdf = f.read()
        
        # mode; Speed, Distance, Manipulation1, Manipulation2
        self.ik_solver = TRAC_IK("base",
                        "left_gripper",
                        urdf,
                        0.005,  # default seconds
                        1e-5,  # default epsilon
                        "Speed")

        self.old_state = np.zeros((6,))
        self.max_num_steps = 50
        print("INIT DONE!")
      

    ## gym methods

    def reset_model(self):
        print("last state:",self.old_state)
        print("New Episode!")
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        ## random target location
        qpos[-2:] = qpos[-2:] + self.np_random.uniform(low=-0.15, high=0.15, size=2)
        ## random box location
        qpos[8:10] = qpos[8:10] + self.np_random.uniform(low=-0.15, high=0.15, size=2)

        self.set_state(qpos, qvel)

        target_pos = np.array([0.6 , 0.3 , 0.15])
        target_quat = np.array([1.0, 0.0 , 0.0, 0])
        target = np.concatenate((target_pos, target_quat))
        action_jt_space = self.do_ik(ee_target= target, jt_pos = self.data.qpos[1:8].flat)
        if action_jt_space is not None:
            self.apply_action(action_jt_space)
        ## for calculating velocities
        # self.old_state = np.zeros((6,))
        self.contacted = False
        self.out_of_bound = 0
        self.num_step = 0
        curr_state = self._get_obs()
        print("start state:",curr_state)
        return self._get_obs()

    def viewer_setup(self):
        # cam_pos = np.array([0.1, 0.0, 0.7, 0.01, -45., 0.])
        cam_pos = np.array([1.0, 0.0, 0.7, 0.5, -45, 180])
        self.set_cam_position(self.viewer, cam_pos)

    def _get_obs(self):
        ee_x, ee_y = self.data.site_xpos[0][:2]
        box_x, box_y = self.data.site_xpos[1][:2]
        target_x, target_y = self.data.site_xpos[2][:2]

        state = np.array([ee_x, ee_y, box_x, box_y, target_x, target_y])
        vel = (state - self.old_state)/self.dt

        self.old_state = state.copy()
        return state
        

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model._compute_subtree() #pylint: disable=W0212
        self.model.forward()

    ## my methods
    def apply_action(self, action):  
        ctrl = self.data.ctrl.copy()
        # print(ctrl.shape)
        ctrl[:7,0] = np.array(action)
        self.data.ctrl = ctrl
        self.do_simulation(ctrl, 1000)
    
    def do_ik(self, ee_target, jt_pos):
        
        # print("starting to do ik")
        # print(ee_target[:3])
       
        # Populate seed with current angles if not provided
        init_pose_trac = tracik.DoubleVector()
        for i in range(7):
            init_pose_trac.push_back(jt_pos[i])

        x,y,z,qx,qy,qz,qw = ee_target
        qout = list(self.ik_solver.CartToJnt(init_pose_trac, x,y,z,qx,qy,qz,qw ))

        if(len(qout)>0):
            # print("ik sol:",qout)
            return qout
        else:
            print("!no result found")
            return None
        

    def close_gripper(self, left_gap=0):
        pass

    def set_cam_position(self, viewer, cam_pos):
        for i in range(3):
            viewer.cam.lookat[i] = cam_pos[i]
        viewer.cam.distance = cam_pos[3]
        viewer.cam.elevation = cam_pos[4]
        viewer.cam.azimuth = cam_pos[5]
        viewer.cam.trackbodyid = -1



    def _step(self, action):


        ## hack for the init of mujoco.env
        if(action.shape[0]>2):
            return np.zeros((6,1)), 0, False, {}
        
        self.num_step += 1
        old_action_jt_space = self.data.qpos[1:8].T.copy()

        ## parsing of primitive actions
        delta_x, delta_y = action
        # print("delta x:%.4f, y:%.4f"%(delta_x, delta_y))
        x, y = self.old_state[:2].copy()
        # print("old x:%.4f, y:%.4f"%(x,y))
        x += delta_x*0.05
        y += delta_y*0.05
        # print("x:%.4f, y:%.4f"%(x,y))
        # print("x:%.4f,y:%.4f"%(0.2*x + 0.6 , 0.3*y + 0.3))
        
        out_of_bound = (x<0.4 or x>0.8) or (y<0.0 or y>0.6)


        if np.abs(delta_x*0.05)>0.0001 or np.abs(delta_y*0.05)>0.0001:
            target_pos = np.array([x , y , 0.15])
            target_quat = np.array([1.0, 0.0 , 0.0, 0])
            target = np.concatenate((target_pos, target_quat))
            action_jt_space = self.do_ik(ee_target= target, jt_pos = self.data.qpos[1:8].flat)
            if (action_jt_space is not None) and (not out_of_bound):
                # print("ik:", action_jt_space)
                self.apply_action(action_jt_space)
            else:
                action_jt_space = old_action_jt_space.copy()

        else:
            action_jt_space = old_action_jt_space.copy()

        # print("controller:",self.data.qpos[1:8].T)
        ## getting state
        ob = self._get_obs()
        gripper_pose = ob[:2]
        box_pose = ob[2:4]
        target_pose = ob[4:6]
        #print("new gripper_pose", gripper_pose, "block pose:", box_pose)
        
        ## reward function definition
        w = [0.1, 1., 0.01, 1., -1e-1, -1e-3, -1,-1]
        
        reward_grip_box = - np.linalg.norm(box_pose- gripper_pose)
        reward_box_target = - np.linalg.norm(box_pose- target_pose)
        reward_first_contact = (np.linalg.norm(box_pose- gripper_pose) < 0.05) and (not self.contacted)
        
        angle_bw_target_block = atan2(target_pose[1] - box_pose[1], target_pose[0] - box_pose[0])
        angle_bw_block_gripper = atan2(box_pose[1] - gripper_pose[1], box_pose[0] - gripper_pose[0])
        reward_angle = abs((angle_bw_target_block - angle_bw_block_gripper) - pi)

        if(reward_first_contact==1):
            self.contacted = True
        reward_reaching_goal = np.linalg.norm(box_pose- target_pose) < 0.02             #assume: my robot has 2cm error

        total_reward = w[1]*reward_box_target \
                        + w[0]*reward_grip_box   \
                        + w[6] * out_of_bound \
                        + w[5] * np.square(action).sum()  \
                            + w[2]*reward_first_contact \
                            + w[3]*reward_reaching_goal \
                            + w[7]*reward_angle
                               
        box_x, box_y, box_z = self.data.site_xpos[1]
        info = {   
                "r_grip_box": reward_grip_box, \
                "r_box_target" :  reward_box_target, \
                "r_contact" : reward_first_contact, 
                # "r_reached" : reward_reaching_goal  \
                }
        if reward_reaching_goal == 1:
            done = True
            info["done"] = "goal reached"
        elif box_z < -0.02 or box_x<0.4 or box_x >0.8 or box_y <0.0 or box_y >0.6 :
            done = True
            info["done"] = "box out of bounds"
        elif (self.num_step > self.max_num_steps):
            done = True
            info["done"] = "max_steps_reached"
        else: 
            done = False

        info['absolute_ob'] = ob.copy()
        relative_ob = ob
        relative_ob[4:] -= relative_ob[2:4]
        relative_ob[2:4] -= relative_ob[:2]

        return relative_ob, total_reward, done, info
                                        
    def apply_hindsight(self, states, actions, goal_state):
        '''generates hindsight rollout based on the goal
        '''
        goal = goal_state[2:4]
        her_states, her_rewards = [], []
        for i in range(len(actions)):
            state = states[i]
            state[-2:] = goal
            reward = self.calc_reward(state, goal, actions[i])
            her_states.append(state)
            her_rewards.append(reward)

        goal_state[-2:] = goal
        her_states.append(goal_state)

        return her_states, her_rewards
    
    def calc_reward(self, state, goal, action):
        
        ## parsing of primitive actions
        delta_x, delta_y = action
        
        x, y = self.old_state[:2].copy()
        x += delta_x*0.05
        y += delta_y*0.05
        
        out_of_bound = (x<0.4 or x>0.8) or (y<0.0 or y>0.6)

        gripper_pose = state[:2]
        box_pose = state[2:4]
        target_pose = goal
        
        ## reward function definition
        w = [0.1, 1., 0.01, 1., -1e-1, -1e-3, -1, -1]
        reward_grip_box = - np.linalg.norm(box_pose- gripper_pose)
        reward_box_target = - np.linalg.norm(box_pose- target_pose)
        reward_first_contact = (np.linalg.norm(box_pose- gripper_pose) < 0.05) and (not self.contacted)
        reward_reaching_goal = np.linalg.norm(box_pose- target_pose) < 0.02             #assume: my robot has 2cm error
        
        angle_bw_target_block = atan2(target_pose[1] - box_pose[1], target_pose[0] - box_pose[0])
        angle_bw_block_gripper = atan2(box_pose[1] - gripper_pose[1], box_pose[0] - gripper_pose[0])
        reward_angle = abs((angle_bw_target_block - angle_bw_block_gripper) - pi)

        total_reward = w[1]*reward_box_target \
                        + w[0]*reward_grip_box   \
                        + w[6] * out_of_bound \
                        + w[5] * np.square(action).sum()  \
                            + w[2]*reward_first_contact \
                            + w[3]*reward_reaching_goal \
                            + w[7]*reward_angle
                            
                            
        return total_reward



if __name__ == "__main__":
    
    from ipdb import set_trace

    env = BaxterEnv()
    EVAL_EPISODE = 10
    reward_mat = []

    try:

        for l in range(EVAL_EPISODE):
            print("Evaluating:%d"%(l+1))
            done = False
            i =0
            random_r = 0
            ob = env.reset()
            print(ob)
            while((not done) and (i<1000)):
                
                ee_x, ee_y = env.data.site_xpos[0][:2]
                box_x, box_y = env.data.site_xpos[1][:2]
                # action = np.array([(box_x - ee_x), (box_y - ee_y)])
                action = env.action_space.sample()
                ob, reward, done, info = env.step(action)
                # print(i, action, ob, reward)
                # print(i, ob, reward, info)
                # print( i, done)    
                i+=1
                sleep(.01)
                env.render()
                random_r += reward


                # set_trace()

            print("num steps:%d, total_reward:%.4f"%(i+1, random_r))
            reward_mat += [random_r]
        print("reward - mean:%f, var:%f"%(np.mean(reward_mat), np.var(reward_mat)))
    except KeyboardInterrupt:
        print("Exiting!")
