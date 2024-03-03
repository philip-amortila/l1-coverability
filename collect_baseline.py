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

torch.backends.cudnn.enabled = False

args = base_utils.get_args()
Policy = CartEntropyPolicy

#mu-based coverability objective 
def mu_objective(average_occ_sa,mu,c):
    return 1 / (average_occ_sa + c * mu) #since mu is uniform, this is just multiplying by 396
    #return mu / (average_occ_sa + c * mu)

#linearly rescales the reward function to be between 0 and 1
def reward_shaping(reward_fn):
    r_max = np.max(reward_fn)
    r_min = np.min(reward_fn)
    new_reward = reward_fn
    new_reward -= r_min
    new_reward /= (r_max - r_min)
    return new_reward

def number_unique_states(average_occ_sa):
    return np.count_nonzero(average_occ_sa)

#measuring l1-cov for different values of epsilon
def l1_cov(average_occ_sa,mu,eps,c):
    return 1 / (average_occ_sa + eps * c * mu) #since mu is uniform, this is just multiplying by 396
    #return mu / (average_occ_sa + c * mu)

#maxent reward fn 
def grad_ent(pt):
    if args.grad_ent:
        grad_p = -np.log(pt)
        grad_p[grad_p > 100] = 1000
        return grad_p
    eps = 1/np.sqrt(base_utils.total_state_space)
    return 1/(pt + eps)

# Get the initial zero-state for the env.
def init_state(env):
    if env == "Pendulum-v0":
        return [np.pi, 0] 
    elif env == "MountainCarContinuous-v0":
        return [-0.50, 0]

def collect_entropy_policies(env, epochs, T, MODEL_DIR, measurements='el'):

    #parse which measurements to make
    measure_entropy = 'e' in measurements
    measure_l1_cov = 'l' in measurements

    ent_reward_fn = np.zeros(shape=(tuple(base_utils.num_states)))
    cov_reward_fn = np.zeros(shape=(tuple(base_utils.num_sa)))

    # set initial state to base, motionless state. 
    seed = []
    if args.env == "Pendulum-v0":
        env.env.state = [np.pi, 0]
        seed = env.env._get_obs()
    elif args.env == "MountainCarContinuous-v0":
        env.env.state = [-0.50, 0]
        seed = env.env.state

    #collect the data for one run: avg occupancy, entropy, l1-cov
    cov_running_avg_p = np.zeros(shape=(tuple(base_utils.num_states)))
    cov_running_avg_ent = 0
    cov_running_avg_l1 = 0
    #aggregate in a list for plotting purposes. this list for all the epochs in this run
    cov_running_avg_entropies = []
    cov_running_avg_l1s = []
    cov_running_avg_ps = []

    ent_running_avg_p = np.zeros(shape=(tuple(base_utils.num_states)))
    ent_running_avg_ent = 0
    ent_running_avg_l1 = 0
    ent_running_avg_entropies = []
    ent_running_avg_l1s = [] #size should be 1 \times epochs
    ent_running_avg_ps = []

    baseline_running_avg_p = np.zeros(shape=(tuple(base_utils.num_states)))
    baseline_running_avg_ent = 0
    baseline_running_avg_l1 = 0
    baseline_running_avg_entropies = []
    baseline_running_avg_l1s = []
    baseline_running_avg_ps = []
        
    cov_policies = []
    ent_policies = []
    initial_state = init_state(args.env)

    cov_new_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
    cov_new_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
    ent_new_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
    ent_new_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
    base_new_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
    base_new_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))

    for i in range(epochs):

        #phil: filename for this epoch
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

        #estimate occupancy for random policy
        p_baseline, p_sa_baseline, _  = cov_policy.execute_random(T, zero_reward, num_rollouts=args.num_rollouts, initial_state=initial_state) 
        #estimate occupancy for cov policy
        new_cov_p, new_cov_psa, _ = cov_policy.execute(T, reward_fn=zero_reward, num_rollouts =args.num_rollouts, initial_state=initial_state)
        #estimate occupancy for ent policy 
        new_ent_p, new_ent_psa, _ = ent_policy.execute(T, reward_fn=zero_reward, num_rollouts =args.num_rollouts,initial_state=initial_state)
            
        #this is the mixture occupancies for defining the reward functions
        cov_new_average_p = cov_new_average_p * (i)/float(i+1) + new_cov_p/float(i+1)
        cov_new_average_psa = cov_new_average_psa * (i)/float(i+1) + new_cov_psa/float(i+1)
        ent_new_average_p = ent_new_average_p * (i)/float(i+1) + new_ent_p/float(i+1)
        ent_new_average_psa = ent_new_average_psa * (i)/float(i+1) + new_ent_psa/float(i+1)
        base_new_average_p = base_new_average_p * (i)/float(i+1) + p_baseline/float(i+1)
        base_new_average_psa = base_new_average_psa * (i)/float(i+1) + p_sa_baseline/float(i+1)

        #re-estimate the occupancies for evaluation:
        cov_eval_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
        cov_eval_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
        ent_eval_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
        ent_eval_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
        base_eval_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
        base_eval_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))

        #re-estimate the occupancy for every policy and average them. 
        for j in range(len(cov_policies)):
            cov_eval_i_p, cov_eval_i_psa,_ = cov_policies[j].execute(T, reward_fn=zero_reward, num_rollouts =args.num_rollouts, initial_state=initial_state)
            ent_eval_i_p, ent_eval_i_psa,_ = ent_policies[j].execute(T, reward_fn=zero_reward, num_rollouts =args.num_rollouts,initial_state=initial_state)
            cov_eval_average_p = cov_eval_average_p * (j)/float(j+1) + cov_eval_i_p/float(j+1)
            cov_eval_average_psa = cov_eval_average_psa * (j)/float(j+1) + cov_eval_i_psa/float(j+1)
            ent_eval_average_p = ent_eval_average_p * (j)/float(j+1) + ent_eval_i_p/float(j+1)
            ent_eval_average_psa = ent_eval_average_psa * (j)/float(j+1) + ent_eval_i_psa/float(j+1)

        base_eval_average_p, base_eval_average_psa,_  = cov_policy.execute_random(T, zero_reward, num_rollouts=args.num_rollouts, initial_state=initial_state) 

        #calculate the new entropy:
        if measure_entropy:
            cov_eval_average_ent = scipy.stats.entropy(cov_eval_average_p.flatten())
            ent_eval_average_ent = scipy.stats.entropy(ent_eval_average_p.flatten())
            round_entropy_baseline = scipy.stats.entropy(base_eval_average_p.flatten())
        else:
            cov_eval_average_ent = 0
            ent_eval_average_ent = 0
            round_entropy_baseline = 0

        #calculating the l1-coverability values
        mu = np.ones(shape=tuple(base_utils.num_sa)) #tabular coverability distribution
        mu *= 1/(np.prod(base_utils.num_sa)) #n_sa is a tuple e.g. (for mountaincar, [12,11,3]). multiply to get number of state-actions pairs
        eps = args.reg_eps #epsilon
        print('eps:', eps)
        c_inf = np.prod(base_utils.num_sa)

        if measure_l1_cov:
            #measuring l_1 coverability. this is fairly expensive, use the args.measurements flag to avoid this
            #reward functions to be maximized
            l1_cov_reward_fn_ent = l1_cov(ent_eval_average_psa, mu, eps, c_inf) 
            l1_cov_reward_fn_ent = reward_shaping(l1_cov_reward_fn_ent)
            l1_cov_reward_fn_cov = l1_cov(cov_eval_average_psa, mu, eps, c_inf) 
            l1_cov_reward_fn_cov = reward_shaping(l1_cov_reward_fn_cov)
            l1_cov_reward_fn_base = l1_cov(base_eval_average_psa, mu, eps, c_inf)
            l1_cov_reward_fn_base = reward_shaping(l1_cov_reward_fn_base)

            #create new policy, optimize the l1 cov objective using the entropy policy cover
            measurement_policy_ent = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim) 
            measurement_policy_ent.learn_policy(l1_cov_reward_fn_ent, det_initial_state=False, sa_reward=True,
                    initial_state=initial_state, 
                    episodes=args.episodes, 
                    train_steps=args.train_steps)
            #what l1-cov reward does it get? 
            _, _, ent_l1 = measurement_policy_ent.execute(T, l1_cov_reward_fn_ent, num_rollouts=args.num_rollouts, initial_state=initial_state)

            measurement_policy_cov = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim) 
            measurement_policy_cov.learn_policy(l1_cov_reward_fn_cov, det_initial_state=False, sa_reward=True,
                initial_state=initial_state, 
                episodes=args.episodes, 
                train_steps=args.train_steps)
            _, _, cov_l1 = measurement_policy_cov.execute(T, l1_cov_reward_fn_cov,num_rollouts=args.num_rollouts, initial_state=initial_state)

            measurement_policy_base = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim) 
            measurement_policy_base.learn_policy(l1_cov_reward_fn_base, det_initial_state=False, sa_reward=True,
                initial_state=initial_state, 
                episodes=args.episodes, 
                train_steps=args.train_steps)
            _, _, base_l1 = measurement_policy_base.execute(T, l1_cov_reward_fn_base,num_rollouts=args.num_rollouts, initial_state=initial_state)

        else:
            cov_l1 = 0
            ent_l1 = 0
            base_l1 = 0

        #new reward fns for next round:

        #reward function for maxent
        ent_reward_fn = grad_ent(ent_new_average_p) 

        #reward function for coverability algorithm
        #using a regularization parameter 
        regularization_eps = args.reg_eps
        print('regularization_eps:', regularization_eps)
        cov_reward_fn = l1_cov(cov_new_average_psa,mu,regularization_eps,c_inf)
        #rescaling the rewards to lie in [0,1]
        cov_reward_fn = reward_shaping(cov_reward_fn)
        
        # If in pendulum, set velocity to 0 with some probability
        # if args.env == "Pendulum-v0" and random.random() < 0.3:
        #     initial_state[1] = 0

        # Update experimental running averages.
        cov_eval_average_number_sa = number_unique_states(cov_eval_average_psa)
        cov_eval_average_number_s = number_unique_states(cov_eval_average_p)
        cov_running_avg_ent = cov_running_avg_ent * (i)/float(i+1) + cov_eval_average_ent/float(i+1)
        cov_running_avg_entropies.append(cov_eval_average_ent)
        cov_running_avg_l1s.append(cov_l1)
        cov_running_avg_ps.append(cov_eval_average_p)  

        # # Update entropy running averages.
        ent_eval_average_number_sa = number_unique_states(ent_eval_average_psa)
        ent_eval_average_number_s = number_unique_states(ent_eval_average_p)
        ent_running_avg_ent = ent_running_avg_ent * (i)/float(i+1) + ent_eval_average_ent/float(i+1)
        ent_running_avg_entropies.append(ent_eval_average_ent)
        ent_running_avg_l1s.append(ent_l1)
        ent_running_avg_ps.append(ent_eval_average_p)

        # # Update baseline running averages.
        base_eval_average_number_sa = number_unique_states(base_eval_average_psa)
        base_eval_average_number_s = number_unique_states(base_eval_average_p)
        baseline_running_avg_ent = baseline_running_avg_ent * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        baseline_running_avg_entropies.append(round_entropy_baseline)
        baseline_running_avg_l1s.append(base_l1)
        baseline_running_avg_ps.append(base_eval_average_p) 

        #save the data and the models
        #models:
        print('saving models to filename:', MODEL_DIR + "cov_policy"+"_"+str(i))
        torch.save(cov_policy,MODEL_DIR + "cov_policy"+"_"+str(i))
        torch.save(ent_policy,MODEL_DIR + "ent_policy"+"_"+str(i))
        #data: 
        print('saving data to filename:', filename)
        file = open(filename,'wb')
        #dump the entropies, l1_cov, occupancies, number of unique states, and number of unique state-actins pairs for all three algorithms
        data = [cov_eval_average_ent, ent_eval_average_ent, round_entropy_baseline]
        data += [cov_l1, ent_l1, base_l1]
        data += [cov_eval_average_p, ent_eval_average_p, base_eval_average_p]
        data += [cov_eval_average_number_s, ent_eval_average_number_s, base_eval_average_number_s]
        data += [cov_eval_average_number_sa, ent_eval_average_number_sa, base_eval_average_number_sa]
        pickle.dump(data, file)
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

        print("cov_entropy[%d] = %f" % (i, cov_eval_average_ent))
        print("cov_running_avg_ent = %s" % cov_running_avg_ent)
        print("cov average unique states = %s" % cov_eval_average_number_s)
        print("cov average unique state-actions = %s" % cov_eval_average_number_sa)
        print("cov_l1[%d] = %f" % (i, cov_l1))
        print("cov_running_avg_l1 = %s" % cov_running_avg_l1)

        print("..........")

        print("ent_entropy[%d] = %f" % (i, ent_eval_average_ent))
        print("ent_running_avg_ent = %s" % ent_running_avg_ent)
        print("ent average unique states = %s" % ent_eval_average_number_s)
        print("ent average unique state-actions = %s" % ent_eval_average_number_sa)
        print("ent_l1[%d] = %f" % (i, ent_l1))
        print("ent_running_avg_l1 = %s" % ent_running_avg_l1)
        

        print("..........")

        print("round_entropy_baseline[%d] = %f" % (i, round_entropy_baseline))
        print("running_avg_ent_baseline = %s" % baseline_running_avg_ent)
        print("base average unique states = %s" % base_eval_average_number_s)
        print("base average unique state-actions = %s" % base_eval_average_number_sa)
        print("base_l1[%d] = %f" % (i, base_l1))
        print("baseline_running_avg_l1 = %s" % baseline_running_avg_l1)

        print("--------------------------------")
    
    #end of epochs    
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

    MODEL_DIR = 'models-' + args.env + '/models_' + str(args.exp_name)+'_' + str(args.replicate) + '_' + TIME + '/'

    if args.save_models:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        # save metadata from the run. 
        with open(MODEL_DIR + "metadata", "w") as metadata:
            metadata.write("args: %s\n" % args)
            metadata.write("num_states: %s\n" % str(base_utils.num_states))
            metadata.write("state_bins: %s\n" % base_utils.state_bins)

    cov_policies, ent_policies = collect_entropy_policies(env, args.epochs, args.T, MODEL_DIR,args.measurements) 
    env.close()

    print("DONE")

if __name__ == "__main__":
    main()


