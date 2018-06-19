import tensorflow as tf
import tensorflow.contrib as tc

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)

    # add summary op
    tf.summary.histogram("actions_discrete_pred", y)
    tf.summary.histogram("actions_discrete_pred_onehot", y_hard)

    y = tf.stop_gradient(y_hard - y) + y
  return y

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, discrete_action_size, cts_action_size, name='actor', layer_norm=True, select_action = True, hidden_unit_list=None):
        super(Actor, self).__init__(name=name)
        self.discrete_action_size = discrete_action_size
        self.cts_action_size = cts_action_size
        self.layer_norm = layer_norm
        self.select_action = select_action
        self.hidden_unit_list = hidden_unit_list

    def __call__(self, obs, reuse=False, num_layers=3, hidden_units=64, temperature = 1.):
        
        hidden_unit_list = self.hidden_unit_list

        if hidden_unit_list is None:
            hidden_unit_list = [hidden_units]*num_layers

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            for hidden_neurons in hidden_unit_list:
                x = tf.layers.dense(x, hidden_neurons)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
             
            x_discrete = tf.layers.dense(x, self.discrete_action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            
            if self.select_action:
                x_discrete = gumbel_softmax(logits=x_discrete, temperature=temperature, hard= True)

            x_cts = tf.layers.dense(x, self.cts_action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            
            # [-1,1]
            x_cts = tf.nn.tanh(x_cts)

            x = tf.concat(values=[x_discrete, x_cts], axis=-1)
            ## for the sake of quick recalling
            y = tf.identity(x, name="preactivation")
            
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True, hidden_unit_list=None):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.hidden_unit_list = hidden_unit_list

    def __call__(self, obs, action, reuse=False, num_layers=3, hidden_units=64):
       
        hidden_unit_list = self.hidden_unit_list

        if hidden_unit_list is None:
            hidden_unit_list = [hidden_units]*num_layers

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, hidden_unit_list[0])
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)

            for hidden_neurons in hidden_unit_list[1:]:
                x = tf.layers.dense(x, hidden_neurons)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class NewCritic(Model):
    def __init__(self, name='critic', layer_norm=True, hidden_unit_list=None):
        super(NewCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.hidden_unit_list = hidden_unit_list

    def __call__(self, obs, action, reuse=False, num_layers=3, hidden_units=64):
        hidden_unit_list = self.hidden_unit_list
        
        if hidden_unit_list is None:
            hidden_unit_list = [hidden_units]*num_layers

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.concat([obs, action], axis=-1)


            for hidden_neurons in hidden_unit_list:
                x = tf.layers.dense(x, hidden_neurons)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
