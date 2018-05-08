import glob
import tensorflow as tf
import os.path as osp
import numpy as np
import os

from HER.ddpg.models import Actor, Critic
from HER import logger
from HER.common.mpi_running_mean_std import RunningMeanStd
from HER.ddpg.ddpg import normalize
import HER.common.tf_util as U
from HER.ddpg.util import read_checkpoint_local
from HER.successor_prediction_model.v1.models import classifier

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
    def num_params(self):
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

    def termination(self, obs, primitive_id):
        return self.skillset[primitive_id].termination(obs)

    def get_terminal_state_from_memory(self, primitive_id, obs, primitive_params):
        return self.skillset[primitive_id].get_terminal_state_from_memory(obs = obs, primitive_params = primitive_params)

    def get_critic_value(self, primitive_id, obs, primitive_params):
        return self.skillset[primitive_id].get_critic_value(obs = obs, primitive_params = primitive_params)

    def get_prob_skill_success(self, primitive_id, obs, primitive_params):
        return self.skillset[primitive_id].get_prob_skill_success(obs = obs, primitive_params = primitive_params)

    def num_skill_params(self, primitive_id):
        return self.skillset[primitive_id].num_params


def mirror(*args, **kwargs):
    if 'obs' in kwargs:
        return kwargs['obs']
    else:
        return args[0]

class DDPGSkill(object):
    def __init__(self, observation_shape=(1,), normalize_observations=True, observation_range=(-5., 5.), 
        action_range=(-1., 1.), nb_actions=3, layer_norm = True, skill_name = None, restore_path=None,
        action_func = None, obs_func = None, num_params=None, termination = None, get_full_state_func=None):
        
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        
        # Parameters.
        self.skill_name = skill_name
        self.restore_path = osp.expanduser(restore_path)
        self.normalize_observations = normalize_observations
        self.action_range = action_range
        self.observation_range = observation_range
        self.actor = Actor(nb_actions=nb_actions, name= "%s/actor"%skill_name, layer_norm=layer_norm)
        self.critic = Critic(layer_norm=layer_norm, name= "%s/critic"%skill_name)
        
        self.successor_prob_model = classifier(in_shape = observation_shape[0], 
                                            out_shape = 1,
                                            name = "%s/suc_pred_model"%skill_name, sess=None,
                                            log_dir=None, train = False, in_tensor = self.obs0)

        self.num_params = num_params
        # load memory
        print("searching for memory in %s"%osp.join(self.restore_path, 'memory'))
        memory_filename = glob.glob(osp.join(self.restore_path, 'memory' , '*.csv'))[0]

        self.memory = np.loadtxt(memory_filename , delimiter= ',')
        self.starting_state_goal = self.memory[:, :observation_shape[0]]
        self.ending_state = self.memory[:, observation_shape[0]:]

        if termination: 
            self.termination = termination
        else:
            self.termination = lambda x: False

        # funcs
        self.get_action = action_func if action_func is not None else mirror
        self.get_obs = obs_func if obs_func is not None else mirror
        self.get_full_state = get_full_state_func if get_full_state_func is not None else mirror
        
        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('%s/obs_rms'%skill_name):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None
        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        
        self.actor_tf = self.actor(normalized_obs0)
        self.critic_tf = self.critic(normalized_obs0, self.actor_tf)
        self.success_prob = self.successor_prob_model.prob
        ## loader and saver
        self.loader_ddpg = tf.train.Saver(self.create_restore_var_dict())
        self.loader_successor_model = tf.train.Saver(self.create_restore_var_dict_successor_model())
        
    
    def create_restore_var_dict_successor_model(self):
        var_restore_dict_successor_model = {}

        model_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='%s/suc_pred_model'%self.skill_name)

        for var in model_var:
            name = var.name
            name = name.replace("%s/"%self.skill_name,"")
            var_restore_dict_successor_model[name[:-2]] = var

        logger.info("restoring following pred model vars\n"+"-"*20)
        logger.info("num of vars to restore:%d"%len(var_restore_dict_successor_model))
        logger.info(str(var_restore_dict_successor_model))
        logger.info("-"*50)

        return var_restore_dict_successor_model

    
    def create_restore_var_dict(self):

        var_restore_dict_ddpg={}

        for var in self.actor.trainable_vars:
            name = var.name
            name = name.replace("%s/"%self.skill_name, "")
            var_restore_dict_ddpg[name[:-2]] = var
        
        obs_rms_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s/obs_rms'%self.skill_name)

        for var in obs_rms_var:
            name = var.name
            name = name.replace("%s/"%self.skill_name,"")
            var_restore_dict_ddpg[name[:-2]] = var

        for var in self.critic.trainable_vars:
            name = var.name
            name = name.replace("%s/"%self.skill_name, "")
            var_restore_dict_ddpg[name[:-2]] = var
        
        logger.info("restoring following ddpg vars\n"+"-"*20)
        logger.info("num of vars to restore:%d"%len(var_restore_dict_ddpg))
        logger.info(str(var_restore_dict_ddpg))
        logger.info("-"*50)

        return var_restore_dict_ddpg

    def restore_skill(self, path, sess):
        self.sess = sess
        
        
        print('Restore path : ',path)
        model_checkpoint_path = read_checkpoint_local(osp.join(path, "model"))
        if model_checkpoint_path:
            self.loader_ddpg.restore(U.get_session(), model_checkpoint_path)
            logger.info("Successfully loaded %s skill"%self.skill_name)

        model_checkpoint_path = read_checkpoint_local(osp.join(path, "pred_model"))
        if model_checkpoint_path:
            self.loader_successor_model.restore(U.get_session(), model_checkpoint_path)
            logger.info("Successfully loaded pred model for %s skill"%self.skill_name)




    def pi(self, obs, primitive_params,compute_Q=False):
        
        actor_tf = self.actor_tf
        feed_dict = {self.obs0: [self.get_obs(obs=obs, params=primitive_params)]}
        
        action = self.sess.run(actor_tf, feed_dict=feed_dict)
        action = action.flatten()
        action = np.clip(action, -1, 1)
        return self.get_action(action, obs)


    def get_prob_skill_success(self, obs, primitive_params):
        feed_dict = {self.obs0: [self.get_obs(obs=obs, params=primitive_params)]}
        success_prob = self.sess.run(self.success_prob, feed_dict)
        return success_prob

    def get_critic_value(self, obs, primitive_params):
        feed_dict = {self.obs0: [self.get_obs(obs=obs, params=primitive_params)]}
        # print("skill %s"%self.skill_name)
        q_value = self.sess.run(self.critic_tf, feed_dict)
        return q_value

    def get_terminal_state_from_memory(self, obs, primitive_params):
        skill_obs = self.get_obs(obs = obs, params = primitive_params)

        # do Nearest neighbour 
        min_dist_idx = np.argmin( np.linalg.norm(self.starting_state_goal - skill_obs, axis=1))
        next_obs = self.ending_state[min_dist_idx].copy()

        # append target
        target = obs[-3:]
        next_state = np.concatenate((next_obs, target))
        next_full_state = self.get_full_state(next_state, prev_obs = obs)

        return next_full_state



if __name__ == "__main__":
    from HER.skills import set1
    import numpy as np

    with U.single_threaded_session() as sess:
        sset = SkillSet(set10.skillset)
        sess.run(tf.global_variables_initializer())
        
        obs = np.random.rand(set10.skillset[0]['observation_shape'][0])
        sset.restore_skillset(sess)
        action,q = sset.pi(primitive_id=0, obs=obs)
        print(action.shape)
