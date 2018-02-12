from HER.ddpg.models import Actor
import tensorflow as tf
from HER import logger
from HER.common.mpi_running_mean_std import RunningMeanStd
from HER.ddpg.ddpg import normalize
import HER.common.tf_util as U
import os.path as osp

from ipdb import set_trace

class SkillSet:
    def __init__(self, skills):
        self.skillset = []
        for skill in skills:
            self.skillset.append(DDPGSkill(**skill))

    def restore_skillset(self, sess):
        for skill in self.skillset:
            skill.restore_skill(path = skill.restore_path, sess = sess)

    def pi(self, obs, primitive_id=0):
        return self.skillset[primitive_id].pi(obs)


class DDPGSkill(object):
    def __init__(self, observation_shape=(1,), normalize_observations=True, observation_range=(-5., 5.), 
        action_range=(-1., 1.), nb_actions=3, layer_norm = True, skill_name = None, restore_path=None):
        
        # Debug info log
        # logger.info("skill params")
        # logger.info(str(locals()))
        # logger.info("-"*20)
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        
        # Parameters.
        self.skill_name = skill_name
        self.restore_path = restore_path
        self.normalize_observations = normalize_observations
        self.action_range = action_range
        self.observation_range = observation_range
        self.actor = Actor(nb_actions=nb_actions, name=skill_name, layer_norm=layer_norm)
        
        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None
        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        
        self.actor_tf = self.actor(normalized_obs0)
    
    def restore_skill(self, path, sess):
        self.sess = sess
        train_vars = self.actor.trainable_vars
        var_restore_dict={}
        for var in train_vars:
            name = var.name
            name = name.replace(self.skill_name, "actor")
            var_restore_dict[name[:-2]] = var

        # print("restoring following vars\n"+"-"*20)
        # print(var_restore_dict)
        # print("-"*50)
        loader = tf.train.Saver(var_restore_dict)
        print('Restore path : ',path)
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            model_checkpoint_path = osp.join(path, osp.basename(checkpoint.model_checkpoint_path))
            loader.restore(U.get_session(), model_checkpoint_path)
            print("Successfully loaded %s skill"%self.skill_name)

    def pi(self, obs, compute_Q=False):
        
        actor_tf = self.actor_tf
        feed_dict = {self.obs0: [obs]}
        
        action = self.sess.run(actor_tf, feed_dict=feed_dict)
        q = None
        action = action.flatten()
        action = np.clip(action, -1, 1)
        return action, q


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