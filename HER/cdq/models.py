import tensorflow as tf
import tensorflow.contrib as tc


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
    def __init__(self, nb_actions, name='actor', layer_norm=True, hidden_unit_list=None):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_unit_list = hidden_unit_list

    def __call__(self, obs, reuse=False, num_layers=3, hidden_units=64):
        hidden_unit_list = self.hidden_unit_list

        if hidden_unit_list is None:
            hidden_unit_list = [hidden_units]*num_layers

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            for hidden_neuron in hidden_unit_list:
                x = tf.layers.dense(x, hidden_neuron)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
             
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            
            ## for the sake of quick recalling
            y = tf.identity(x, name="preactivation")
            
            x = tf.nn.tanh(y)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True, hidden_unit_list = None):
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

            for hidden_neuron in hidden_unit_list[1:]:
                x = tf.layers.dense(x, hidden_neuron)
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

