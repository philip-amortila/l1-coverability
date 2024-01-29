# experimenting with curiosity exploration method.
# Code derived from: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

# example command setting args in base_utils.py
# python curiosity.py  --models_dir=models-MountainCarContinuous-v0/models_2018_11_28-17-45/ --env="MountainCarContinuous-v0"
# python curiosity.py  --models_dir=models-Pendulum-v0/models_2018_11_29-09-48/ --env="Pendulum-v0"

import os
import sys
import time

import copy

import random
import numpy as np
import scipy.stats
import gym
from gym import wrappers

import torch
from torch.distributions import Categorical

import base_utils
args = base_utils.get_args()

#phil: this only works for 3 actions
def select_action(probs):
    m = Categorical(probs)
    action = m.sample()
    if (action.item() == 1):
        return [0]
    elif (action.item() == 0):
        return [-1]
    return [1]

def get_obs(state):
    if base_utils.args.env == "Pendulum-v0":
        theta, thetadot = state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
    elif base_utils.args.env == "MountainCarContinuous-v0":
        return np.array(state)

    # unroll for T steps and compute p
def execute_policy_internal(env, T, policies, state, render):
    random_T = np.floor(random.random()*T)
    p = np.zeros(shape=(tuple(base_utils.num_states)))
    p_sa = np.zeros(shape=(tuple(base_utils.num_sa)))
    random_initial_state = []
    #commit to a policy for T steps
    idx = np.random.randint(0,high=len(policies))
    pi = policies[idx]

    #phil: changed this
    for t in range(T):
        # Compute average probability over action space for state.
        probs = torch.tensor(np.zeros(shape=(1,base_utils.action_dim))).float()
        probs = pi.get_probs(state)
        #probs = torch.tensor(np.zeros(shape=(1,base_utils.action_dim))).float()
        #var = torch.tensor(np.zeros(shape=(1,base_utils.action_dim))).float()
        #for policy in policies:
         #   prob = policy.get_probs(state)
          #  probs += prob
        #probs /= len(policies)
        action = select_action(probs)
        
        state, reward, done, _ = env.step(action)
        p[tuple(base_utils.discretize_state(state))] += 1
        next_probs = torch.tensor(np.zeros(shape=(1,base_utils.action_dim))).float()
        next_probs = pi.get_probs(state)
        next_action = select_action(next_probs) 
        tmp = copy.deepcopy(base_utils.discretize_state(state))
        tmp.append(next_action[0])
        #print('tmp in execute_policy_internal:', tmp)
        p_sa[tuple(tmp)] += 1 #here!

        if (t == random_T and not render):
            random_initial_state = env.env.state

        if render:
            env.render()
        if done:
            break 

    p /= float(T)
    p_sa /= float(T)
    return p, p_sa, random_initial_state

# run a simulation to see how the average policy behaves.
def execute_average_policy(env, policies, T, initial_state=[], avg_runs=1, render=False):
    
    average_p = np.zeros(shape=(tuple(base_utils.num_states)))
    average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
    avg_entropy = 0
    random_initial_state = []

    last_run = avg_runs - 1 #phil: this doesn't get used
    for i in range(avg_runs):
        if len(initial_state) == 0:
            initial_state = env.reset()

        env.env.reset_state = initial_state
        state = env.reset()

        p, p_sa, random_initial_state = execute_policy_internal(env, T, policies, state, False)
        average_p += p
        average_psa += p_sa
        avg_entropy += scipy.stats.entropy(average_p.flatten())

    env.close()
    average_p /= float(avg_runs)
    average_psa /= float(avg_runs)
    avg_entropy /= float(avg_runs) # running average of the entropy 
    entropy_of_final = scipy.stats.entropy(average_p.flatten())

    return average_p, average_psa, avg_entropy, random_initial_state



