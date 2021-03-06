import numpy as np
from random import random

class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, action):
        return action + np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class EpsilonNormalActionNoise(ActionNoise):
    ''' Based on Hindsight Experience Replay paper: https://pdfs.semanticscholar.org/9734/9dee55ba13067f467695eecb3a3bb68e43bd.pdf
    '''
    def __init__(self, mu, sigma, epsilon):
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, action):
        if(random()>self.epsilon):
            return action + np.random.normal(self.mu, self.sigma)
        else:
            return np.random.uniform(-1. , 1., size=action.shape)

    def __repr__(self):
        return 'EpsilonNormalActionNoise(mu={}, sigma={}, epsilon={})'.format(self.mu, self.sigma, self.epsilon)

class EpsilonNormalParameterizedActionNoise(ActionNoise):
    ''' Based on Hindsight Experience Replay paper: https://pdfs.semanticscholar.org/9734/9dee55ba13067f467695eecb3a3bb68e43bd.pdf
    '''
    def __init__(self, mu, sigma, epsilon, discrete_actions_dim):
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
        self.discrete_actions_dim = discrete_actions_dim
        
    def __call__(self, action):
        discrete_actions_prob = action[:self.discrete_actions_dim]
        continuous_actions = action[self.discrete_actions_dim:]
            
        if(random()>self.epsilon):
            continuous_actions += np.random.normal(self.mu, self.sigma)
            action = np.concatenate((discrete_actions_prob, continuous_actions))
            return action
        else:
            discrete_actions_prob = np.zeros((self.discrete_actions_dim,))
            chosen_action = np.random.choice(self.discrete_actions_dim)
            discrete_actions_prob[chosen_action] = 1.0
            continuous_actions = np.random.uniform(-1. , 1., size= action.size - self.discrete_actions_dim)
            action = np.concatenate((discrete_actions_prob, continuous_actions))
            return action

    def __repr__(self):
        return 'EpsilonNormalParameterizedActionNoise(mu={}, sigma={}, epsilon={})'.format(self.mu, self.sigma, self.epsilon)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self, action):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return (action + x)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
