# Collect entropy-based reward policies.

# Changed from using all-1 reward to init to one-hot at: 2018_11_30-10-00

# python collect_baseline.py --env="MountainCarContinuous-v0" --T=200 --train_steps=400 --episodes=300 --epochs=50 --exp_name=test

import sys
import os
home_dir = os.getenv('HOME')
sys.path = ['/Users/philipamortila/Documents/GitHub/coverability_experiments_final'+'/gym-fork'] + sys.path #phil: change this to your local folder

import time
from datetime import datetime
import logging

import copy

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import UnivariateSpline #changed this
from scipy.stats import norm

import gym

from cart_entropy_policy import CartEntropyPolicy
import base_utils
import curiosity
import plotting

import torch
from torch.distributions import Normal
import random

import pickle

from itertools import islice

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def moving_averages(values, size):
    for selection in window(values, size):
        yield sum(selection) / size

args = base_utils.get_args()
Policy = CartEntropyPolicy

#pushforward-based coverability objective
def push_objective(average_occ,transition_matrix):
    return None #not implemented yet

#mu-based coverability objective (for planning algo)
def mu_objective(average_occ_sa,mu,c):
    return 1 / (average_occ_sa + c * mu) #multiplying by 396
    #return mu / (average_occ_sa + c * mu)

#rescales the reward function between 0 and 1
def reward_shaping(reward_fn):
    r_max = np.max(reward_fn)
    r_min = np.min(reward_fn)
    new_reward = reward_fn
    new_reward -= r_min
    new_reward /= (r_max - r_min)
    #print(new_reward)
    return new_reward
    #return mu / (average_occ_sa + c * mu)

def reward_shaping_polynomial(reward_fn):
    r_max = np.max(reward_fn)
    r_min = np.min(reward_fn)
    new_reward = reward_fn
    new_reward -= r_min
    new_reward /= (r_max - r_min)
    #print(new_reward)
    return new_reward**10000
    #return mu / (average_occ_sa + c * mu)

def number_unique_states(average_occ_sa):
    return np.count_nonzero(average_occ_sa)


#measuring l1 cov for different values of epsilon
def l1_cov(average_occ_sa,mu,eps,c):
    return 1 / (average_occ_sa + eps * c * mu) #multiplying by 396
    #return mu / (average_occ_sa + c * mu)

#maxent reward fn 
def grad_ent(pt):
    if args.grad_ent:
        grad_p = -np.log(pt)
        grad_p[grad_p > 100] = 1000
        return grad_p
    eps = 1/np.sqrt(base_utils.total_state_space)
    return 1/(pt + eps)

def online_rewards(average_p, average_ps, t):
    eps = 1/np.sqrt(base_utils.total_state_space)
    reward_fn = np.zeros(shape=average_p.shape)
    for ap in average_ps:
        reward_fn += 1/(ap + eps)
    reward_fn += np.sqrt(t)*average_p
    return reward_fn

# Get the initial zero-state for the env.
def init_state(env):
    if env == "Pendulum-v0":
        return [np.pi, 0] 
    elif env == "MountainCarContinuous-v0":
        return [-0.50, 0]

#same as collect_entropy_policies_many_runs but for one run only 
def collect_entropy_policies(env, epochs, T, MODEL_DIR, measurements='elp'):

    #parse which measurements to make
    measure_entropy = 'e' in measurements
    measure_l1_cov = 'l' in measurements
    measure_pg = 'p' in measurements

    video_dir = 'videos/' + args.exp_name

    ent_reward_fn = np.zeros(shape=(tuple(base_utils.num_states)))
    cov_reward_fn = np.zeros(shape=(tuple(base_utils.num_sa)))

    # set initial state to base, motionless state. #PHIL: i don't think this works.
    seed = []
    if args.env == "Pendulum-v0":
        env.env.state = [np.pi, 0]
        seed = env.env._get_obs()
    elif args.env == "MountainCarContinuous-v0":
        env.env.state = [-0.50, 0]
        seed = env.env.state

    #phil: need aggregated run data
    #phil: these lists aggregate the (epoch,entropy) data from each run, for coverability, entropy, and baselines respectively
    #phil: these lists aggregate the (epoch,l1-cov) data from each run, for coverability, entropy, and baselines respectively

    #this is the data for one run
    #running avg occupancy, entropy, l1-cov
    cov_running_avg_p = np.zeros(shape=(tuple(base_utils.num_states)))
    cov_running_avg_ent = 0
    cov_running_avg_l1 = 0
    cov_running_avg_pg = 0
    #aggregate in a list for plotting purposes. this list for all the epochs in this run
    cov_running_avg_entropies = []
    cov_running_avg_l1s = []
    cov_running_avg_pgs = []
    cov_running_avg_ps = []

    ent_running_avg_p = np.zeros(shape=(tuple(base_utils.num_states)))
    ent_running_avg_ent = 0
    ent_running_avg_l1 = 0
    ent_running_avg_pg = 0
    ent_running_avg_entropies = []
    ent_running_avg_l1s = [] #size should be 1 \times epochs
    ent_running_avg_pgs = []
    ent_running_avg_ps = []

    baseline_running_avg_p = np.zeros(shape=(tuple(base_utils.num_states)))
    baseline_running_avg_ent = 0
    baseline_running_avg_l1 = 0
    baseline_running_avg_pg = 0
    baseline_running_avg_entropies = []
    baseline_running_avg_l1s = [] #phil: not implemented yet
    baseline_running_avg_pgs = []
    baseline_running_avg_ps = []
        
    cov_policies = []
    ent_policies = []
    initial_state = init_state(args.env)
    #phil: why different initial states?
    ent_initial_state = init_state(args.env)

    cov_new_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
    cov_new_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
    ent_new_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
    ent_new_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
    base_new_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
    base_new_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))

    cov_pg_scores = []
    ent_pg_scores = []
    base_pg_scores = []
    none_pg_scores = []

    # print('testing density estimation:' )
    # p_baselines = []
    # test_policy=Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim)
    # zero_reward = np.zeros(shape=(tuple(base_utils.num_sa)))
    # norms = []
    # for i in range(10):
    #     p_baseline_1, _, _  = test_policy.execute_random(T, zero_reward, num_rollouts=args.num_rollouts*(1), initial_state=initial_state) 
    #     p_baseline_1 = np.asarray(p_baseline_1)
    #     p_baseline_2,_,_ = test_policy.execute_random(T, zero_reward, num_rollouts=args.num_rollouts*(100), initial_state=initial_state) 
    #     v = (p_baseline_2-p_baseline_1).flatten()
    #     print('v:', v)
    #     print('np.linalg.norm(v,ord=np.inf):',np.linalg.norm(v,ord=np.inf))
    #     norms.append(np.linalg.norm(v,ord=np.inf))

    # print('mean:', np.mean(norms))
    # print('std:', scipy.stats.sem(norms))

    for i in range(epochs):

        #phil: new filename for this epoch
        filename = plotting.FIG_DIR + '/' + args.exp_name + '/' + args.exp_name + '_' + str(args.replicate) + '_' + str(i)

        # Learn policy that maximizes current reward function.
        cov_policy = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim)
        ent_policy = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim) 

        print('-----starting learn_policy in epoch:', i, '------')
        print('initial_state:', initial_state)
        if i == 0:
            cov_policy.learn_policy(cov_reward_fn, det_initial_state=False, sa_reward = True,
                episodes=0, 
                train_steps=0)
            ent_policy.learn_policy(ent_reward_fn, det_initial_state=False, sa_reward = False,
                episodes=0, 
                train_steps=0)
        else:
            cov_policy.learn_policy(cov_reward_fn, det_initial_state=False, sa_reward=True,
                initial_state=initial_state, 
                episodes=args.episodes, 
                train_steps=args.train_steps)
            ent_policy.learn_policy(ent_reward_fn, det_initial_state=False, sa_reward=False, 
                initial_state=initial_state, 
                episodes=args.episodes, 
                train_steps=args.train_steps)

        cov_policies.append(cov_policy)
        ent_policies.append(ent_policy)

        epoch = 'epoch_%02d/' % (i) 
        
        zero_reward = np.zeros(shape=(tuple(base_utils.num_sa)))

        #a = 100 # average over this many rollouts for estimating occupancy. #phil: this a is only used for random baseline, rest of the code its called num_rollouts
        #phil: this still only works for 3 actions (-1,0,1)
        #phil: double check the initial state business
        #phil: execute_random shouldn't be in the Policy class...
        print('-----starting execute_random in epoch:', i, '------')
        print('initial_state:', initial_state)
        p_baseline, p_sa_baseline, _  = cov_policy.execute_random(T, zero_reward, num_rollouts=args.num_rollouts, initial_state=initial_state, 
            render=args.render, video_dir=video_dir+'/baseline/'+epoch) 

        #phil: commenting this, was replaced by num_rollouts in execute_random
        # for av in range(a - 1):
        #     next_p_baseline = cov_policy.execute_random(T, zero_reward)
        #     p_baseline += next_p_baseline
        #     round_entropy_baseline += scipy.stats.entropy(next_p_baseline.flatten())
        # p_baseline /= float(a)
        # round_entropy_baseline /= float(a) # running average of the entropy

        #estimate the occupancy of the new policy, add it to occupancies
        print('-----starting execute in epoch:', i, '------')
        print('initial_state:', initial_state)
        new_cov_p, new_cov_psa, _ = cov_policy.execute(T, reward_fn=zero_reward, num_rollouts =args.num_rollouts, initial_state=initial_state, 
            render=args.render, video_dir=video_dir+'/normal/'+epoch)
        new_ent_p, new_ent_psa, _ = ent_policy.execute(T, reward_fn=zero_reward, num_rollouts =args.num_rollouts,initial_state=initial_state, 
            render=args.render, video_dir=video_dir+'/online/'+epoch)
            

        #this is the mixture occupancies for training
        cov_new_average_p = cov_new_average_p * (i)/float(i+1) + new_cov_p/float(i+1)
        cov_new_average_psa = cov_new_average_psa * (i)/float(i+1) + new_cov_psa/float(i+1)
        ent_new_average_p = ent_new_average_p * (i)/float(i+1) + new_ent_p/float(i+1)
        ent_new_average_psa = ent_new_average_psa * (i)/float(i+1) + new_ent_psa/float(i+1)
        base_new_average_p = base_new_average_p * (i)/float(i+1) + p_baseline/float(i+1)
        base_new_average_psa = base_new_average_psa * (i)/float(i+1) + p_sa_baseline/float(i+1)


        # print('new_cov_psa:', new_cov_psa)
        # print('cov_new_average_psa:', cov_new_average_psa)

        #re-estimate the occupancies for evaluation:
        cov_eval_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
        cov_eval_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
        ent_eval_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
        ent_eval_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
        base_eval_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
        base_eval_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))

        #re-estimate the occupancy for every policy and average them. this might be overkill (can sample policy uniformly at random)
        #commenting this out now
        # print('starting eval occupancy estimation')
        # for j in range(len(cov_policies)):
        #     cov_eval_i_p, cov_eval_i_psa,_ = cov_policies[j].execute(T, reward_fn=zero_reward, num_rollouts =args.num_rollouts, initial_state=initial_state, 
        #     render=args.render, video_dir=video_dir+'/normal/'+epoch)
        #     ent_eval_i_p, ent_eval_i_psa,_ = ent_policies[j].execute(T, reward_fn=zero_reward, num_rollouts =args.num_rollouts,initial_state=initial_state, 
        #     render=args.render, video_dir=video_dir+'/online/'+epoch)
        #     cov_eval_average_p = cov_eval_average_p * (j)/float(j+1) + cov_eval_i_p/float(j+1)
        #     cov_eval_average_psa = cov_eval_average_psa * (j)/float(j+1) + cov_eval_i_psa/float(j+1)
        #     ent_eval_average_p = ent_eval_average_p * (j)/float(j+1) + ent_eval_i_p/float(j+1)
        #     ent_eval_average_psa = ent_eval_average_psa * (j)/float(j+1) + ent_eval_i_psa/float(j+1)

        # base_eval_average_p, base_eval_average_psa,_  = cov_policy.execute_random(T, zero_reward, num_rollouts=args.num_rollouts, initial_state=initial_state, 
        #     render=args.render, video_dir=video_dir+'/baseline/'+epoch) 

        #print('cov_eval_average_p:', cov_eval_average_p)

        #calculate the new entropy:
        if measure_entropy:
            print('calculating entropies')
            # cov_new_average_ent = scipy.stats.entropy(cov_eval_average_p.flatten())
            # ent_new_average_ent = scipy.stats.entropy(ent_eval_average_p.flatten())
            # round_entropy_baseline = scipy.stats.entropy(base_eval_average_p.flatten())
            cov_new_average_ent = scipy.stats.entropy(cov_new_average_p.flatten())
            ent_new_average_ent = scipy.stats.entropy(ent_new_average_p.flatten())
            round_entropy_baseline = scipy.stats.entropy(base_new_average_p.flatten())
        else:
            cov_new_average_ent = 0
            ent_new_average_ent = 0
            round_entropy_baseline = 0

        #calculating the l1-coverability values
        mu = np.ones(shape=tuple(base_utils.num_sa)) #tabular coverability distribution
        mu *= 1/(np.prod(base_utils.num_sa)) #n_sa is a tuple e.g. ([12,11,3]), multiply to get number of state-actions pairs
        eps = 1/float(args.episodes)
        c_inf = np.prod(base_utils.num_sa)

        if measure_l1_cov and i%5==0:
        #if measure_l1_cov:
            print('starting l1_cov measurements at epoch', i)
            l1_cov_reward_fn_ent = l1_cov(ent_eval_average_psa, mu, eps, c_inf) 
            l1_cov_reward_fn_cov = l1_cov(cov_eval_average_psa, mu, eps, c_inf) 
            l1_cov_reward_fn_base = l1_cov(base_eval_average_psa, mu, eps, c_inf)

            # print('-----debugging the l1_cov reward function------', )
            # print('l1_cov_reward_fn_cov in collect_entropy_policies:', l1_cov_reward_fn_cov)
            # print('cov_new_average_psa in collect_entropy_policies:', cov_eval_average_psa)
            # print('eps:', eps)
            # print('cov_new_average_psa sums to 1:', np.sum(cov_eval_average_psa))

            #phil: maximize these 
            #phil: create new policy, optimize the l1 cov objective using the entropy policy cover
            measurement_policy_ent = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim) 
            measurement_policy_ent.learn_policy(l1_cov_reward_fn_ent, det_initial_state=False, sa_reward=True,
                    initial_state=initial_state, 
                    episodes=args.episodes, 
                    train_steps=args.train_steps)
            #phil: what l1-cov reward does it get? can also see where it went
            _, _, ent_l1 = measurement_policy_ent.execute(T, l1_cov_reward_fn_ent, num_rollouts=args.num_rollouts, initial_state=initial_state, 
                render=args.render, video_dir=video_dir+'/normal/'+epoch)

            #phil: do the same for cov policy. optimize the l1 cov objective for the cov policy cover
            #phil: technically this can be measured by the next iteration so this is a bit inefficient
            measurement_policy_cov = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim) 
            measurement_policy_cov.learn_policy(l1_cov_reward_fn_cov, det_initial_state=False, sa_reward=True,
                initial_state=initial_state, 
                episodes=args.episodes, 
                train_steps=args.train_steps)
            _, _, cov_l1 = measurement_policy_cov.execute(T, l1_cov_reward_fn_cov,num_rollouts=args.num_rollouts, initial_state=initial_state, 
                render=args.render, video_dir=video_dir+'/normal/'+epoch)

            #phil: last for base policy.
            measurement_policy_base = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim) 
            measurement_policy_base.learn_policy(l1_cov_reward_fn_base, det_initial_state=False, sa_reward=True,
                initial_state=initial_state, 
                episodes=args.episodes, 
                train_steps=args.train_steps)
            _, _, base_l1 = measurement_policy_base.execute(T, l1_cov_reward_fn_base,num_rollouts=args.num_rollouts, initial_state=initial_state, 
                render=args.render, video_dir=video_dir+'/normal/'+epoch)

        else:
            cov_l1 = 0
            ent_l1 = 0
            base_l1 = 0

        #phil: testing the reset dist
        # print('attempting init_state_reset...')
        # cov_policy.init_state_reset(cov_policies,T)

        # print('attempting learn_policy_reset...')
        # #    def learn_policy_reset(self, policies, reward_fn, sa_reward=True, true_reward=False,
        # # episodes=1000, train_steps=1000, start_steps=10000):
        # cov_policy.learn_policy_reset(cov_policies, T, zero_reward, true_reward=True, 
        #         episodes = args.episodes, 
        #         train_steps = args.train_steps)

        print('epoch i', i)
        if measure_pg and i % 5 == 0:
            print('MEASURING PG')
            rf_cov_pg = 0
            rf_ent_pg = 0
            rf_base_pg = 0
            rf_none_pg = 0 #policy gradient loss using true initial state distribution
            #phil: measure the PG loss on a "hard" reward function
            #phil: right now using the environment's true reward function

            goal_reward = np.zeros(shape=(tuple(base_utils.num_states)))
            goal_reward[11,:] = 100
            #print(goal_reward)
            
            # #"reward-free" policy for cov
            rf_policy_cov = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim)
            #optimize it with reset function
            # #    def learn_policy _reset(self, policies, reward_fn, sa_reward=True,episodes=1000, train_steps=1000, start_steps=10000):
            rf_policy_cov.learn_policy_reset(cov_policies,T,goal_reward,sa_reward=False, true_reward=False,
                             episodes=args.episodes,
                             train_steps=args.train_steps)
            # # what reward does it get?
            _, _, rf_cov_pg = rf_policy_cov.execute(T, goal_reward,sa_reward=False,true_reward=False,num_rollouts=args.num_rollouts, initial_state=initial_state, 
                 render=args.render, video_dir=video_dir+'/normal/'+epoch)

            # #"reward-free" policy for ent
            # rf_policy_ent = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim)
            # #optimize it with reset function
            # #    def learn_policy_reset(self, policies, reward_fn, sa_reward=True,episodes=1000, train_steps=1000, start_steps=10000):
            # rf_policy_ent.learn_policy_reset(ent_policies,T,goal_reward,sa_reward=False, true_reward=False,
            #                 episodes=args.episodes,
            #                 train_steps=args.train_steps)
            # # what reward does it get?
            # _, _, rf_ent_pg =rf_policy_ent.execute(T, goal_reward,sa_reward=False, true_reward=False,num_rollouts=args.num_rollouts, initial_state=initial_state, 
            #     render=args.render, video_dir=video_dir+'/normal/'+epoch)

            # #"reward-free" policy for base
            # rf_policy_base = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim)
            # rf_policy_base.learn_policy_reset_random(T,goal_reward,sa_reward=False,true_reward=False,
            #                 episodes=args.episodes,
            #                 train_steps=args.train_steps)
            # # what reward does it get?
            # _, _, rf_base_pg =rf_policy_base.execute(T, goal_reward,sa_reward=False,true_reward=False,num_rollouts=args.num_rollouts, initial_state=initial_state, 
            #     render=args.render, video_dir=video_dir+'/normal/'+epoch) 
            print('optimizing with none reset')
            rf_policy_none = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim)
            rf_policy_none.learn_policy(goal_reward, sa_reward=False, true_reward=False,
                initial_state=[], 
                episodes=args.episodes, 
                train_steps=args.train_steps)
            _, _, rf_none_pg = rf_policy_none.execute(T, goal_reward,sa_reward=False,true_reward=False,num_rollouts=args.num_rollouts, initial_state=initial_state, 
                render=args.render, video_dir=video_dir+'/normal/'+epoch) 

            # cov_pg_scores.append(rf_cov_pg)#print('rf_cov_pg, rf_ent_pg, rf_base_pg:', rf_cov_pg, rf_ent_pg, rf_base_pg)
            # ent_pg_scores.append(rf_ent_pg)
            # base_pg_scores.append(rf_base_pg)
            none_pg_scores.append(rf_none_pg)

        else:
            rf_cov_pg = 0
            rf_ent_pg = 0
            rf_base_pg = 0
            rf_none_pg = 0

        #new reward fns for next round:
        #reward function for maxent
        ent_reward_fn = grad_ent(ent_new_average_p) 
        #reward function for coverability algorithm
        # print('testing reward shaping')
        #phil: trying the epsilon thing:
        cov_reward_fn = l1_cov(cov_new_average_psa,mu,0.00001,c_inf)
        #print('cov_reward_fn:', cov_reward_fn)
        cov_reward_fn = reward_shaping(cov_reward_fn)
        #print('new cov_reward_fn:', cov_reward_fn)

        #cov_reward_fn = mu_objective(cov_new_average_psa, mu, c_inf) 
        #cov_reward_fn = reward_shaping_polynomial(cov_reward_fn)
        


        
        
#         #measuring l1 cov for different values of epsilon
# def l1_cov(average_occ_sa,mu,eps,c):
#     return 1 / (average_occ_sa + eps * c * mu) #multiplying by 396
#     #return mu / (average_occ_sa + c * mu)
        

        
        # Force first round to be equal     #01/18: commenting this out.
        # if i == 0:
        #     cov_average_p = p_baseline
        #     ent_average_p = p_baseline
        #     #phil: not doing this for p_sa, maybe I need to?
        #     cov_round_avg_ent = round_entropy_baseline
        #     ent_round_avg_ent = round_entropy_baseline

        # If in pendulum, set velocity to 0 with some probability
        if args.env == "Pendulum-v0" and random.random() < 0.3:
            initial_state[1] = 0

        # Update experimental running averages.
        #phil: creating number of states statistics
        cov_eval_average_number_sa = number_unique_states(cov_new_average_psa)
        cov_eval_average_number_s = number_unique_states(cov_new_average_p)
        cov_running_avg_ent = cov_running_avg_ent * (i)/float(i+1) + cov_new_average_ent/float(i+1)
        # cov_running_avg_l1 = cov_running_avg_l1 * (i)/float(i+1) + cov_l1/float(i+1)
        # cov_running_avg_pg = cov_running_avg_pg * (i)/float(i+1) + rf_cov_pg/float(i+1)
        # cov_running_avg_p = cov_running_avg_p * (i)/float(i+1) + cov_new_average_p/float(i+1)
        # cov_running_avg_entropies.append(cov_running_avg_ent)
        # cov_running_avg_l1s.append(cov_running_avg_l1)
        # cov_running_avg_pgs.append(cov_running_avg_pg)
        # cov_running_avg_ps.append(cov_running_avg_p)  

        # # Update entropy running averages.
        ent_eval_average_number_sa = number_unique_states(ent_new_average_psa)
        ent_eval_average_number_s = number_unique_states(ent_new_average_p)
        ent_running_avg_ent = ent_running_avg_ent * (i)/float(i+1) + ent_new_average_ent/float(i+1)
        # ent_running_avg_l1 = ent_running_avg_l1 * (i)/float(i+1) + ent_l1/float(i+1)
        # ent_running_avg_pg = ent_running_avg_pg * (i)/float(i+1) + rf_ent_pg/float(i+1)
        # ent_running_avg_p = ent_running_avg_p * (i)/float(i+1) + ent_new_average_p/float(i+1)
        # ent_running_avg_entropies.append(ent_running_avg_ent)
        # ent_running_avg_l1s.append(ent_running_avg_l1)
        # ent_running_avg_pgs.append(ent_running_avg_pg)
        # ent_running_avg_ps.append(ent_running_avg_p)  

        # # Update baseline running averages.
        base_eval_average_number_sa = number_unique_states(base_new_average_psa)
        base_eval_average_number_s = number_unique_states(base_new_average_p)
        baseline_running_avg_ent = baseline_running_avg_ent * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        # baseline_running_avg_l1 = baseline_running_avg_l1 * (i)/float(i+1) + base_l1/float(i+1)
        # baseline_running_avg_pg = baseline_running_avg_pg * (i)/float(i+1) + rf_base_pg/float(i+1)
        # baseline_running_avg_p = baseline_running_avg_p * (i)/float(i+1) + p_baseline/float(i+1)
        # baseline_running_avg_entropies.append(baseline_running_avg_ent)
        # baseline_running_avg_l1s.append(baseline_running_avg_l1)
        # baseline_running_avg_ps.append(baseline_running_avg_p) 

        #phil: this version only plots the entropies per round (instead of the running average entropies)
        # cov_running_avg_ent = cov_running_avg_ent * (i)/float(i+1) + cov_new_average_ent/float(i+1)
        # cov_running_avg_l1 = cov_running_avg_l1 * (i)/float(i+1) + cov_l1/float(i+1)
        # cov_running_avg_pg = cov_running_avg_pg * (i)/float(i+1) + rf_cov_pg/float(i+1)
        # cov_running_avg_p = cov_running_avg_p * (i)/float(i+1) + cov_new_average_p/float(i+1)
        cov_running_avg_entropies.append(cov_new_average_ent)
        cov_running_avg_l1s.append(cov_l1)
        cov_running_avg_pgs.append(rf_cov_pg)
        cov_running_avg_ps.append(cov_new_average_p)  

        # Update entropy running averages.
        # ent_running_avg_ent = ent_running_avg_ent * (i)/float(i+1) + ent_new_average_ent/float(i+1)
        # ent_running_avg_l1 = ent_running_avg_l1 * (i)/float(i+1) + ent_l1/float(i+1)
        # ent_running_avg_pg = ent_running_avg_pg * (i)/float(i+1) + rf_ent_pg/float(i+1)
        # ent_running_avg_p = ent_running_avg_p * (i)/float(i+1) + ent_new_average_p/float(i+1)
        ent_running_avg_entropies.append(ent_new_average_ent)
        ent_running_avg_l1s.append(ent_l1)
        ent_running_avg_pgs.append(rf_ent_pg)
        ent_running_avg_ps.append(ent_new_average_p)

        # Update baseline running averages.
        # baseline_running_avg_ent = baseline_running_avg_ent * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        # baseline_running_avg_l1 = baseline_running_avg_l1 * (i)/float(i+1) + base_l1/float(i+1)
        # baseline_running_avg_pg = baseline_running_avg_pg * (i)/float(i+1) + rf_base_pg/float(i+1)
        # baseline_running_avg_p = baseline_running_avg_p * (i)/float(i+1) + p_baseline/float(i+1)
        baseline_running_avg_entropies.append(round_entropy_baseline)
        baseline_running_avg_l1s.append(base_l1)
        baseline_running_avg_pgs.append(rf_base_pg)
        baseline_running_avg_ps.append(p_baseline) 

        #SAVE THE DATA, MODELS
        #models:
        print('saving models to filename:', MODEL_DIR + "cov_policy"+"_"+str(i))
        torch.save(cov_policy,MODEL_DIR + "cov_policy"+"_"+str(i))
        torch.save(ent_policy,MODEL_DIR + "ent_policy"+"_"+str(i))
        #data: 
        print('saving data to filename:', filename)
        file = open(filename,'wb')
        #dump the entropies and l1_cov values for all three algorithms
        data = [cov_new_average_ent, ent_new_average_ent, round_entropy_baseline, cov_l1, ent_l1, base_l1]
        data.append(rf_cov_pg)
        data.append(rf_ent_pg)
        data.append(rf_base_pg)
        data.append(rf_none_pg)
        data.append(cov_eval_average_p)
        data.append(ent_eval_average_p)
        data.append(base_eval_average_p)
        data.append(cov_eval_average_number_sa)
        data.append(ent_eval_average_number_sa)
        data.append(base_eval_average_number_sa)
        pickle.dump(data, file)
        # data_occupancies = [cov_running_avg_ps, ent_running_avg_ps, baseline_running_avg_ps]
        # print(data_occupancies[0])
        # pickle.dump(data_occupancies, file)
        file.close()

        print("--------------------------------")
        print("cov_p=")
        print(new_cov_p)

        print("cov_average_p =") 
        print(cov_new_average_p)

        print("ent_p=")
        print(new_ent_p)

        print("ent_average_p")
        print(ent_new_average_p)

        print("base_p=")
        print(p_baseline)

        print("base_new_average_p")
        print(base_new_average_p)

        print("---------------------")

        print("cov_ent[%d] = %f" % (i, cov_new_average_ent))
        print("cov_running_avg_ent = %s" % cov_running_avg_ent)
        print("cov average unique states = %s" % cov_eval_average_number_s)
        print("cov average unique state-actions = %s" % cov_eval_average_number_sa)
        print("cov_l1[%d] = %f" % (i, cov_l1))
        #print("cov_running_avg_l1 = %s" % cov_running_avg_l1)
        print("rf_cov_pg[%d] = %f" % (i, rf_cov_pg))
        #print("cov_running_avg_pg = %s" % cov_running_avg_pg)

        print("..........")

        print("ent_round_avg_ent[%d] = %f" % (i, ent_new_average_ent))
        print("ent_running_avg_ent = %s" % ent_running_avg_ent)
        print("ent average unique states = %s" % ent_eval_average_number_s)
        print("ent average unique state-actions = %s" % ent_eval_average_number_sa)
        print("ent_l1[%d] = %f" % (i, ent_l1))
        #print("ent_running_avg_l1 = %s" % ent_running_avg_l1)
        print("rf_ent_pg[%d] = %f" % (i, rf_ent_pg))
        #print("ent_running_avg_pg = %s" % ent_running_avg_pg)

        print("..........")

        print("round_entropy_baseline[%d] = %f" % (i, round_entropy_baseline))
        print("running_avg_ent_baseline = %s" % baseline_running_avg_ent)
        print("base average unique states = %s" % base_eval_average_number_s)
        print("base average unique state-actions = %s" % base_eval_average_number_sa)
        print("base_l1[%d] = %f" % (i, base_l1))
        #print("baseline_running_avg_l1 = %s" % baseline_running_avg_l1)
        print("rf_base_pg[%d] = %f" % (i, rf_base_pg))
        #print("baseline_running_avg_pg = %s" % baseline_running_avg_pg)

        print("--------------------------------")
    
    #end of epochs
    #instead of plotting, dump it to pickle file
    
    #filename = str(args.exp_name) + '_' + str(args.replicate)
    
    return cov_policies, ent_policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)

    # Make environment.
    env = gym.make(args.env)
    # TODO: limit acceleration (maybe also speed?) for Pendulum.
    if args.env == "Pendulum-v0":
        env.env.max_speed = 8
        env.env.max_torque = 1
    env.seed(int(time.time())) # seed environment

    TIME = datetime.now().strftime('%Y_%m_%d-%H-%M')
    #MODEL_DIR = 'models-' + args.env + '/models_' + TIME + '/'

    MODEL_DIR = 'models-' + args.env + '/models_' + str(args.exp_name)+'_' + str(args.replicate) + '_' + TIME + '/'

    print(MODEL_DIR)

    if args.save_models:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        # save metadata from the run. 
        with open(MODEL_DIR + "metadata", "w") as metadata:
            metadata.write("args: %s\n" % args)
            metadata.write("num_states: %s\n" % str(base_utils.num_states))
            metadata.write("state_bins: %s\n" % base_utils.state_bins)

    # test_reward_fn = np.ones(shape=(12,11,3))
    # test_reward_fn[0,0,:] = 1/2
    # test_reward_fn[0,1,:] = 3/4
    # test_reward_fn = reward_shaping(test_reward_fn)

    #phil: commenting this and putting it in the plotting file 
    # plotting.FIG_DIR = 'figs/' + args.env + '/'
    # plotting.model_time = args.exp_name + '/'
    # if not os.path.exists(plotting.FIG_DIR+plotting.model_time):
    #     os.makedirs(plotting.FIG_DIR+plotting.model_time)

    #phil: you don't need to return this...
    #phil: modifying this to include exp_runs
    cov_policies, ent_policies = collect_entropy_policies(env, args.epochs, args.T, MODEL_DIR,args.measurements) 
    env.close()

    #phil: save models


        #phil: putting this in each iteration
        # for i in range(len(cov_policies)):
        #     torch.save(cov_policies[i],MODEL_DIR + "cov_policy"+"_"+str(i))
        #     torch.save(ent_policies[i],MODEL_DIR + "ent_policy"+"_"+str(i))

    print("DONE")

if __name__ == "__main__":
    main()


