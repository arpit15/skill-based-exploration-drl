from HER.ddpg.models import Actor
import tensorflow as tf
from HER import logger
from HER.common.mpi_running_mean_std import RunningMeanStd
from HER.ddpg.ddpg import normalize
import HER.common.tf_util as U
import os.path as osp
import numpy as np
import os

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


        logger.info("Skill set init!\n" + "#"*50)

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
            skill.restore_skill(path = get_home_path(skill.restore_path), sess = sess)

    def pi(self, obs, primitive_params=None, primitive_id=0):
        ## make obs for the skill
        starting_idx = self.params_start_idx[primitive_id]
        curr_skill_params = primitive_params[starting_idx : (starting_idx+self.skillset[primitive_id].num_params)]
        return self.skillset[primitive_id].pi(obs=obs, primitive_params=curr_skill_params)


class DDPGSkill(object):
    def __init__(self, observation_shape=(1,), normalize_observations=True, observation_range=(-5., 5.), 
        action_range=(-1., 1.), nb_actions=3, layer_norm = True, skill_name = None, restore_path=None,
        action_func = None, obs_func = None, num_params=None):
        
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        
        # Parameters.
        self.skill_name = skill_name
        self.restore_path = restore_path
        self.normalize_observations = normalize_observations
        self.action_range = action_range
        self.observation_range = observation_range
        self.actor = Actor(nb_actions=nb_actions, name=skill_name, layer_norm=layer_norm)
        self.num_params = num_params

        # funcs
        self.get_action = action_func
        self.get_obs = obs_func
        
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

        
        logger.info("restoring following vars\n"+"-"*20)
        logger.info("num of vars to restore:%d"%len(train_vars))
        logger.info(str(var_restore_dict))
        logger.info("-"*50)

        return var_restore_dict

    def restore_skill(self, path, sess):
        self.sess = sess
        
        
        print('Restore path : ',path)
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            model_checkpoint_path = osp.join(path, osp.basename(checkpoint.model_checkpoint_path))
            self.loader.restore(U.get_session(), model_checkpoint_path)
            logger.info("Successfully loaded %s skill"%self.skill_name)

    def pi(self, obs, primitive_params,compute_Q=False):
        
        actor_tf = self.actor_tf
        feed_dict = {self.obs0: [self.get_obs(obs=obs, params=primitive_params)]}
        
        action = self.sess.run(actor_tf, feed_dict=feed_dict)
        q = None
        action = action.flatten()
        action = np.clip(action, -1, 1)
        return self.get_action(action), q


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