from mpi4py import MPI
import tensorflow as tf
import os.path as osp
import numpy as np
import os

from HER.ddpg.models import Actor
from HER import logger
from HER.common.mpi_running_mean_std import RunningMeanStd
from HER.ddpg.ddpg import normalize
import HER.common.tf_util as U
from HER.ddpg.util import read_checkpoint_local

def get_home_path(path):
    curr_home_path = os.getenv("HOME")
    return path.replace("$HOME",curr_home_path)

class SkillSet:
    def __init__(self, skills):
        self.skillset = []
        self.params_start_idx = []
        param_idx = 0
        for skill in skills:
            self.params_start_idx.append(param_idx)
            self.skillset.append(DDPGSkill(**skill))
            param_idx += self.skillset[-1].num_params


        if MPI.COMM_WORLD.Get_rank() == 0: logger.info("Skill set init!\n" + "#"*50)

    @property
    def len(self):
        return len(self.skillset)

    @property
    def params(self):
        num_params = 0
        for skill in self.skillset:
            num_params += skill.num_params

        return num_params

    def restore_skillset(self, sess):
        for skill in self.skillset:
            skill.restore_skill(path = osp.expanduser(skill.restore_path), sess = sess)

    def pi(self, obs, primitive_params=None, primitive_id=0):

        ## make obs for the skill
        starting_idx = self.params_start_idx[primitive_id]
        if primitive_params is not None:
            curr_skill_params = primitive_params[starting_idx : (starting_idx+self.skillset[primitive_id].num_params)]
            #print(self.skillset[primitive_id].skill_name, curr_skill_params)
            return self.skillset[primitive_id].pi(obs=obs, primitive_params=curr_skill_params)
        else:
            print(self.skillset[primitive_id].skill_name)
            return self.skillset[primitive_id].pi(obs=obs, primitive_params=None)

    def termination(self, obs, primitive_id, primitive_params):
        if primitive_params is not None:
            starting_idx = self.params_start_idx[primitive_id]
            curr_skill_params = primitive_params[starting_idx : (starting_idx+self.skillset[primitive_id].num_params)]
            return self.skillset[primitive_id].termination(obs, curr_skill_params)
        else:
            NotImplementedError


def mirror(*args, **kwargs):
    if 'obs' in kwargs:
        return kwargs['obs']
    else:
        return args[0]

class DDPGSkill(object):
    def __init__(self, observation_shape=(1,), normalize_observations=True, observation_range=(-5., 5.), 
        action_range=(-1., 1.), nb_actions=3, layer_norm = True, skill_name = None, restore_path=None,
        action_func = None, obs_func = None, num_params=None, termination = None,**kwargs):
        
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        
        # Parameters.
        self.skill_name = skill_name
        self.restore_path = osp.expanduser(restore_path)
        self.normalize_observations = normalize_observations
        self.action_range = action_range
        self.observation_range = observation_range
        self.actor = Actor(nb_actions=nb_actions, name=skill_name, layer_norm=layer_norm)
        self.num_params = num_params
        if termination: 
            self.termination = termination
        else:
            self.termination = lambda x: False

        # funcs
        self.get_action = action_func if action_func is not None else mirror
        self.get_obs = obs_func if obs_func is not None else mirror
        
        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('%s/obs_rms'%skill_name):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None
        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        
        self.actor_tf = self.actor(normalized_obs0)

        ## loader and saver
        self.loader = tf.train.Saver(self.create_restore_var_dict())
        
    
    def create_restore_var_dict(self):
        train_vars = self.actor.trainable_vars + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s/obs_rms'%self.skill_name)

        var_restore_dict={}
        for var in train_vars:
            name = var.name
            
            ## takes care of obs normalization var
            if("obs_rms" in name):
                name = name.replace("%s/"%self.skill_name,"")
            ## takes care of actor weights 
            elif(self.skill_name in name):
                name = name.replace(self.skill_name, "actor")
            
            var_restore_dict[name[:-2]] = var

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.debug("restoring following vars\n"+"-"*20)
            logger.debug("num of vars to restore:%d"%len(train_vars))
            logger.debug(str(var_restore_dict))
            logger.debug("-"*50)

        return var_restore_dict

    def restore_skill(self, path, sess):
        self.sess = sess
        
        
        print('Restore path : ',path)
        # checkpoint = tf.train.get_checkpoint_state(path)
        # if checkpoint and checkpoint.model_checkpoint_path:
        model_checkpoint_path = read_checkpoint_local(osp.join(path, "model"))
        if model_checkpoint_path:
            # model_checkpoint_path = osp.join(path, osp.basename(checkpoint.model_checkpoint_path))
            self.loader.restore(U.get_session(), model_checkpoint_path)
            if MPI.COMM_WORLD.Get_rank() == 0: logger.info("Successfully loaded %s skill"%self.skill_name)

    def pi(self, obs, primitive_params,compute_Q=False):
        
        actor_tf = self.actor_tf
        feed_dict = {self.obs0: [self.get_obs(obs=obs, params=primitive_params)]}
        
        action = self.sess.run(actor_tf, feed_dict=feed_dict)
        action = action.flatten()
        action = np.clip(action, -1, 1)
        return self.get_action(action, obs)


if __name__ == "__main__":
    from HER.skills import set1
    import numpy as np

    with U.single_threaded_session() as sess:
        sset = SkillSet(set1.skillset)
        sess.run(tf.global_variables_initializer())
        
        obs = np.random.rand(set1.skillset[0]['observation_shape'][0])
        sset.restore_skillset(sess)
        action,q = sset.pi(primitive_id=0, obs=obs)
        print(action.shape)
