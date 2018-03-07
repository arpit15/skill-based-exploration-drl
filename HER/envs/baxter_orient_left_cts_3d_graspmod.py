import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from gym import spaces
import os 
import os.path as osp
import signal 

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
    actions: (delta_x, delta_y, delta_z, gap) 5cm push
    starting state: (0.63, 0.2, 0.59, 0.27, 0.55, 0.3)
    max_num_steps = 50
    """
    def __init__(self, max_len=10):
        dirname = os.path.dirname(os.path.abspath(__file__)) 
        mujoco_env.MujocoEnv.__init__(self, os.path.join(dirname, "mjc/baxter_orient_left_cts_with_grippers_modified.xml") , 1)
        utils.EzPickle.__init__(self)

        ## mujoco things
        # task space action space

        low = np.array([-1., -1., -1., -1.])
        high = np.array([1., 1., 1., 1.])

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
        urdf_filename = osp.join(dirname, "urdf", "baxter_modified.urdf")
                
        with open(urdf_filename) as f:
            urdf = f.read()
        
        # mode; Speed, Distance, Manipulation1, Manipulation2
        self.ik_solver = TRAC_IK("base",
                        "left_gripper",
                        urdf,
                        0.005,  # default seconds
                        1e-5,  # default epsilon
                        "Speed")

        self.old_state = np.zeros((9,))
        self.max_num_steps = max_len
        print("INIT DONE!")
      

    ## gym methods

    def reset_model(self):
        # print("last state:",self.old_state)
        # print("New Episode!")
        
        reset_state = self.np_random.uniform()>0.5

        if reset_state:
            grasped_qpos = np.array([  0. ,  1.85833336e-01 ,  1.13869066e-01 , -1.57743078e+00,
        1.87249089e+00 ,  1.67964818e+00 ,  1.57880024e+00 ,  2.23699321e+00,
        2.68245581e-02  ,-2.46668516e-02  , 6.09046750e-01  , 2.73356216e-01,
        2.50803949e-03  , 9.99880925e-01  ,-1.19302114e-02   ,9.78048682e-03,
        3.85171977e-04, 6.09046750e-01 - 0.55, 2.73356216e-01 - 0.3])
        # 3.85171977e-04  ,-1.21567676e-01  , 2.00143005e-01])

            ## random target location
            # grasped_qpos[-3:-1] = grasped_qpos[-3:-1] + self.np_random.uniform(low=-0.15, high=0.15, size=2)
            # grasped_qpos[-1] = grasped_qpos[-1] + self.np_random.uniform(low=0.1, high=0.3, size=1)

            qvel = self.init_qvel
            self.set_state(grasped_qpos, qvel)

        else:
            qpos = self.init_qpos + self.np_random.uniform(low=-.002, high=.002, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.002, high=.002, size=self.model.nv)
            
            ## random box location
            qpos[10:12] = qpos[10:12] + self.np_random.uniform(low=-0.15, high=0.15, size=2)
            
            ## random target location
            qpos[-2:] = qpos[10:12] - np.array([0.55, 0.3])
            # qpos[-3:-1] = qpos[-3:-1] + self.np_random.uniform(low=-0.15, high=0.15, size=2)
            # qpos[-1] = qpos[-1] + self.np_random.uniform(low=0.1, high=0.3, size=1)

            self.set_state(qpos, qvel)

            target_pos = np.array(list(qpos[10:12] +  self.np_random.uniform(low=-0.05, high=0.05, size=2)) + [0.2]) 
            target_quat = np.array([1.0, 0.0 , 0.0, 0])
            target = np.concatenate((target_pos, target_quat))
            action_jt_space = self.do_ik(ee_target= target, jt_pos = self.data.qpos[1:8].flat)
            if action_jt_space is not None:
                self.apply_action(action_jt_space)

            self.close_gripper(gap=1)
        ## for calculating velocities
        # self.old_state = np.zeros((6,))
        
        self.num_step = 0
        ob = self._get_obs()
        gripper_pose = ob[:3]
        box_pose = ob[3:6]
        target_pose = ob[6:9]

        relative_ob = np.concatenate([gripper_pose, box_pose - gripper_pose, target_pose - box_pose ])
        return relative_ob

    def viewer_setup(self):
        # cam_pos = np.array([0.1, 0.0, 0.7, 0.01, -45., 0.])
        cam_pos = np.array([1.0, 0.0, 0.7, 0.5, -45, 180])
        self.set_cam_position(self.viewer, cam_pos)

    def _get_obs(self):
        ee_x, ee_y, ee_z = self.data.site_xpos[0][:3]
        box_x, box_y, box_z = self.data.site_xpos[3][:3]
        target_x, target_y, target_z = self.data.site_xpos[4][:3]

        state = np.array([ee_x, ee_y, ee_z, box_x, box_y, box_z, target_x, target_y, target_z])
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
        # print("cmd",action)
        # print("curr jt pos", self.data.qpos[1:8].T)
        ctrl = self.data.ctrl.copy()
        # print(ctrl.shape)
        ctrl[:7,0] = np.array(action)
        self.data.ctrl = ctrl
        self.do_simulation(ctrl, 1000)
        # print("next jt pos", self.data.qpos[1:8].T)
    
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
        

    def close_gripper(self, gap=0):
        # print("before grip location", self.data.site_xpos[0])
        # qpos = self.data.qpos.copy().flatten()
        # qpos[8] = (gap+1)*0.020833
        # qpos[9] = -(gap+1)*0.020833
        # qvel = self.data.qvel.copy().flatten()
        # print(qpos.shape, qvel.shape)
        # self.set_state(qpos, qvel)
        # print("after grip location", self.data.site_xpos[0])
        # print("before grip", self.data.ctrl)
        ctrl = self.data.ctrl.copy()
        ctrl[:7,0] = self.data.qpos[1:8].flatten()
        ctrl[7,0] = (gap+1)*0.020833
        ctrl[8,0] = -(gap+1)*0.020833
        self.data.ctrl = ctrl

        self.do_simulation(ctrl, 1000)
        # print("qpos", self.data.qpos[1:8].T)
        # print("before grip", self.data.ctrl)

    def set_cam_position(self, viewer, cam_pos):
        for i in range(3):
            viewer.cam.lookat[i] = cam_pos[i]
        viewer.cam.distance = cam_pos[3]
        viewer.cam.elevation = cam_pos[4]
        viewer.cam.azimuth = cam_pos[5]
        viewer.cam.trackbodyid = -1



    def _step(self, action):


        ## hack for the init of mujoco.env
        if(action.shape[0]>4):
            return np.zeros((9,1)), 0, False, {}
        
        self.num_step += 1
        old_action_jt_space = self.data.qpos[1:8].T.copy()

        ## parsing of primitive actions
        delta_x, delta_y, delta_z, gap = action

        # print("grip prev controller:",self.data.qpos[1:8].T)
        self.close_gripper(gap)
        # print("delta x:%.4f, y:%.4f"%(delta_x, delta_y))
        x, y, z = self.old_state[:3].copy()
        # print("old x:%.4f, y:%.4f"%(x,y))

        curr_out_of_bound = (x<0.4 or x>0.8) or (y<0.0 or y>0.6) or (z<0.1 or z>0.5)

        x += delta_x*0.05
        y += delta_y*0.05
        z += delta_z*0.05
        # print("x:%.4f, y:%.4f, z:%.4f"%(x,y,z))
        # print("prev controller:",self.data.qpos[1:8].T)
        # print("x:%.4f,y:%.4f"%(0.2*x + 0.6 , 0.3*y + 0.3))
        
        out_of_bound = (x<0.4 or x>0.8) or (y<0.0 or y>0.6) or (z<0.1 or z>0.5)


        if np.abs(delta_x*0.05)>0.0001 or np.abs(delta_y*0.05)>0.0001 or np.abs(delta_z*0.05)>0.0001:
            target_pos = np.array([x , y , z])
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
        gripper_pose = ob[:3]
        box_pose = ob[3:6]
        target_pose = ob[6:9]
        #print("new gripper_pose", gripper_pose, "block pose:", box_pose)
        
        ## reward function definition
        reward_reaching_goal = np.linalg.norm(box_pose- target_pose) < 0.05             #assume: my robot has 5cm error
        total_reward = -1*(not reward_reaching_goal)

                               
        box_x, box_y, box_z = self.data.site_xpos[3]
        
        info = {}
        if reward_reaching_goal == 1:
            done = True
            info["done"] = "goal reached"
        elif box_z < -0.02 or box_x<0.4 or box_x >0.8 or box_y <0.0 or box_y >0.6 :
            done = True
            info["done"] = "box out of bounds"
            total_reward -= (self.max_num_steps - self.num_step) + 5
        elif curr_out_of_bound:
            ## simulation got weird
            done = True
            info["done"] = "simulation unstable"
            total_reward -= (self.max_num_steps - self.num_step) + 5
        elif (self.num_step > self.max_num_steps):
            done = True
            info["done"] = "max_steps_reached"
        else: 
            done = False

        info['absolute_ob'] = ob.copy()
        relative_ob = np.concatenate([gripper_pose, box_pose - gripper_pose, target_pose - box_pose ])
        return relative_ob, total_reward, done, info
                                        
    def apply_hindsight(self, states, actions, goal_state):
        '''generates hindsight rollout based on the goal
        '''
        goal = states[-1][3:6]  + states[-1][:3]    ## this is the absolute goal location
        her_states, her_rewards = [], []
        for i in range(len(actions)):
            state = states[i]

            gripper_pose = state[:3]
            box_pose = state[3:6] + gripper_pose
            target_pose = state[6:9] + box_pose

            state[-3:] = goal.copy() - box_pose
            reward = self.calc_reward(state, goal, actions[i])
            her_states.append(state)
            her_rewards.append(reward)

        goal_state[-3:] = np.array([0., 0., 0.])
        her_states.append(goal_state)

        return her_states, her_rewards
    
    def calc_reward(self, state, goal, action):
        
        gripper_pose = state[:3]
        box_pose = state[3:6] + gripper_pose
        target_pose = goal + box_pose
        
        ## reward function definition
        reward_reaching_goal = np.linalg.norm(box_pose- target_pose) < 0.05
        total_reward = -1*(not reward_reaching_goal)
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
            # env.close_gripper(gap=-1)
            # print(env.data.qpos[8:10])
            
            action1 = np.array([-0.1, 0., -1., 1.0])
            action2 = np.array([0., 0.4, 0., 0.4])
            action3 = np.array([0., 0., 0., -0.4])
            action4 = np.array([0,0,1,-0.4])
            # print(ob)

            for k in range(10):
                env.render()

            while((not done) and (i<1000)):
                
                # ee_x, ee_y, ee_z = env.data.site_xpos[0][:3]
                # box_x, box_y, box_z = env.data.site_xpos[3][:3]
                # action = np.array([(box_x - ee_x), (box_y - ee_y), (box_z - ee_z), 1.0])
                action = env.action_space.sample()
                # action = np.array([0., 0., 0, 1.0])
                # if(i<=20 and i>5):
                #     action = action1
                # elif(i>20 and i<22):
                #     action = action2
                # elif(i>=22 and i<30):
                #     action = action3
                # elif(i>=30 and i<90):
                #     action = action4
                print(action)
                ob, reward, done, info = env.step(action)
                # if(i==22):
                #     print(env.data.qpos.T)
                # print(i, action, ob, reward)
                # print(i, ob, reward, info)
                # print( i, done)    
                i+=1
                sleep(.001)
                env.render()
                random_r += reward


                # set_trace()

            print("num steps:%d, total_reward:%.4f"%(i+1, random_r))
            reward_mat += [random_r]
        print("reward - mean:%f, var:%f"%(np.mean(reward_mat), np.var(reward_mat)))
    except KeyboardInterrupt:
        print("Exiting!")
