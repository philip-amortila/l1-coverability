import numpy as np
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

import gym
from gym import wrappers
import base_utils
import copy

import gc

# Get the initial zero-state for the env.
def init_state(env):
    if env == "Pendulum-v0":
        return [np.pi, 0] 
    elif env == "MountainCarContinuous-v0":
        return [-0.50, 0]


class CartEntropyPolicy(nn.Module):
    def __init__(self, env, gamma, lr, obs_dim, action_dim):
        super(CartEntropyPolicy, self).__init__()

        self.affine1 = nn.Linear(obs_dim, 128)
        self.middle = nn.Linear(128, 128)
        self.affine2 = nn.Linear(128, action_dim)

        torch.nn.init.xavier_uniform_(self.affine1.weight)
        torch.nn.init.xavier_uniform_(self.middle.weight)
        torch.nn.init.xavier_uniform_(self.affine2.weight)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()

        self.env = env
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.init_state = np.array(init_state(base_utils.args.env))
        self.env.seed(int(time.time())) # seed environment

    def init(self, init_policy):
        print("init to policy")
        self.load_state_dict(init_policy.state_dict())

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.middle(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def get_probs(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        return probs

    def select_action_no_grad(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            probs = self.forward(state)
        m = Categorical(probs)
        #print(probs)
        #print(m)
        action = m.sample()
        #print(action)
        #self.saved_log_probs.append(m.log_prob(action))

        if (action.item() == 1):
            return [0]
        elif (action.item() == 0):
            return [-1]
        return [1]

    def select_action(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        #print(probs)
        #print(m)
        action = m.sample()
        #print(action)
        self.saved_log_probs.append(m.log_prob(action))

        if (action.item() == 1):
            return [0]
        elif (action.item() == 0):
            return [-1]
        return [1]

    def update_policy(self):
        R = 0
        policy_loss = [] #
        rewards = []

        # Get discounted rewards from the episode.
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward.float())

        self.optimizer.zero_grad(set_to_none=True)
        policy_loss = torch.cat(policy_loss).sum() # cost function?
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        #self.optimizer.zero_grad()
        gc.collect()

        return policy_loss

    def get_initial_state(self):
        if base_utils.args.env == "Pendulum-v0":
            self.env.env.state = [np.pi, 0] 
            theta, thetadot = self.env.env.state
            return np.array([np.cos(theta), np.sin(theta), thetadot])
        elif base_utils.args.env == "MountainCarContinuous-v0":
            self.env.env.state = [-0.50, 0]
            return np.array(self.env.env.state)

    def get_obs(self):
        if base_utils.args.env == "Pendulum-v0":
            theta, thetadot = self.env.env.state
            return np.array([np.cos(theta), np.sin(theta), thetadot])
        elif base_utils.args.env == "MountainCarContinuous-v0":
            return np.array(self.env.env.state)

    #phil: get initial state from a reset distribution. 
    #phil: picks a uniform random policy from policies and rolls in for tau steps
    def init_state_reset(self,policies,T):
        state = self.env.reset() #phil: env.reset() or self.init_state?
        #print('initial state:', state)
        #pick a random policy
        idx = np.random.randint(0,high=len(policies))
        pi = policies[idx]
        tau = np.random.randint(0,high=T)
        #print('idx:', idx)
        #print('tau:', tau)
        #roll-in for how tau steps?
        for i in range(tau):
            # probs = torch.tensor(np.zeros(shape=(1,base_utils.action_dim))).float()
            # probs = pi.get_probs(state)
            action = self.select_action(state) #phil: note this only works for 3 actions
            #print('action:', action)
            state, _, done, _ = self.env.step(action)
            #print('next state:', state)
            if done:
                break 

        #phil: need env.close()?
        #print('final state:', state)
        return state

    #phil: learn a policy with from a specific reset distribution given by policies
    def learn_policy_reset(self, policies, T, reward_fn, sa_reward=True, true_reward=False,
        episodes=1000, train_steps=1000, start_steps=10000):

        # if len(initial_state) == 0:
        #     initial_state = self.init_state_reset(policies) #phil: not implemented yet
        # print("init: " + str(initial_state))

        print('starting training in learn_policy_reset')
        # print('true_reward in learn_policy_reset:', true_reward)
        running_reward = 0
        running_loss = 0
        for i_episode in range(episodes):
            # if i_episode % 5 == 0: #phil: comment this eventually. or now. 
            #     self.env.env.reset_state = initial_state
            state = self.init_state_reset(policies,T)
            #print('initial state in episode:', state, i_episode) 
            ep_reward = 0
            finished = False
            for t in range(train_steps):  # Don't infinite loop while learning
                #print(state)
                action = self.select_action(state)
                #print(action)
                if true_reward:
                    #prev_state = copy.deepcopy(state)
                    state, reward, done, _ = self.env.step(action)    
                    # if done:
                    #     print('done inside learn_policy_reset:', reward) 
                    #print('t, prev_state, action, state, reward in learn_policy_reset:', t, prev_state, action, state, reward)              
                else: 
                    #next_action = self.select_action(state)
                    tmp = copy.deepcopy(base_utils.discretize_state(state))
                    tmp.append(action[0])
                    if sa_reward:
                        reward = reward_fn[tuple(tmp)] #reward fn is a fn state-action pairs. applied before.
                        state, _, done, _ = self.env.step(action)
                    else:
                        state, _, done, _ = self.env.step(action)
                        reward = reward_fn[tuple(base_utils.discretize_state(state))] #reward fn is a fn of states. applied after.
                    del tmp
                ep_reward += reward
                self.rewards.append(reward)
                if done:
                    # finished = True
                    # TODO: self.env.env.reset_state = initial_state ? 
                    # print('done inside learn_policy_reset, at episode _ with state _:', i_episode, state)
                    # print('ep_reward inside learn_policy_reset', ep_reward)
                    # print('----^ that should be >= 100 ^----')
                    # print('self.rewards in learn_policy_reset:', self.rewards)
                    state = self.init_state_reset(policies,T)
                    #print('new initial state:', state)
                    #self.env.reset() #phil: change this

            # if finished:
            #     print('end of episode:', i_episode, ep_reward)
            #     print('end of episode:', i_episode, self.rewards)
            running_reward = running_reward * 0.99 + ep_reward * 0.01
            if (i_episode == 0):
                running_reward = ep_reward
            
            loss = self.update_policy()
            running_loss = running_loss * 0.99 + loss*.01

            # Log to console.
            if (i_episode) % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLoss: {:.2f}'.format(
                    i_episode, ep_reward, running_reward, running_loss))

        print('done training in learn_policy_reset')

    def init_state_reset_random(self,T):
        state = self.env.reset() #phil: env.reset() or self.init_state?
        #print('initial state:', state)
        #pick a random time index
        tau = np.random.randint(0,high=T)
        #print('tau:', tau)
        #roll-in for tau steps
        for i in range(tau):
            r = random.random()
            action = -1
            if (r < 1/3.):
                action = 0
            elif r < 2/3.:
                action = 1
            # # probs = torch.tensor(np.zeros(shape=(1,base_utils.action_dim))).float()
            # # probs = pi.get_probs(state)
            # action = self.select_action(state) #phil: note this only works for 3 actions
            #print('action:', action)
            state, _, done, _ = self.env.step([action])
            #print('next state:', state)
            if done:
                break 

        #phil: need env.close()?
        #print('final state:', state)
        return state

    #phil: learn a policy with rollins given by the uniformly random policy
    def learn_policy_reset_random(self, T, reward_fn, sa_reward=True, true_reward=False,
        episodes=1000, train_steps=1000, start_steps=10000):

        # if len(initial_state) == 0:
        #     initial_state = self.init_state_reset(policies) #phil: not implemented yet
        # print("init: " + str(initial_state))

        print('starting training in learn_policy_reset_random')
        #print('true_reward in learn_policy_reset_random:', true_reward)
        running_reward = 0
        running_loss = 0
        for i_episode in range(episodes):
            # if i_episode % 5 == 0: #phil: comment this eventually. or now. 
            #     self.env.env.reset_state = initial_state
            state = self.init_state_reset_random(T)
            #print('initial state in episode:', state, i_episode) 
            ep_reward = 0
            finished = False
            for t in range(train_steps):  # Don't infinite loop while learning
                #print(state)
                action = self.select_action(state) 
                if true_reward:
                    #prev_state = copy.deepcopy(state)
                    state, reward, done, _ = self.env.step(action)
                    # if done:
                    #     print('done inside learn_policy_reset:', reward) 
                    #print('t, prev_state, action, state, reward in learn_policy_reset_random:', t, prev_state, action, state, reward)              
                else: 
                    tmp = copy.deepcopy(base_utils.discretize_state(state))
                    tmp.append(action[0])
                    if sa_reward:
                        reward = reward_fn[tuple(tmp)] #reward fn is a fn state-action pairs. applied before.
                        state, _, done, _ = self.env.step(action)
                    else:
                        state, _, done, _ = self.env.step(action)
                        reward = reward_fn[tuple(base_utils.discretize_state(state))] #reward fn is a fn of states. applied after.
                    del tmp
                    #state, _, done, _ = self.env.step(action)
                    # state, _, done, _ = self.env.step(action)
                    # next_action = self.select_action(state)
                    # tmp = copy.deepcopy(base_utils.discretize_state(state))
                    # tmp.append(next_action[0])
                    # if sa_reward:
                    #     reward = reward_fn[tuple(tmp)] #reward fn is a fn state-action pairs
                    # else:
                    #     reward = reward_fn[tuple(base_utils.discretize_state(state))] #reward fn is a fn of states
                #print(reward)
                ep_reward += reward
                self.rewards.append(reward)
                if done:
                    finished = True
                    # TODO: self.env.env.reset_state = initial_state ? 
                    # print('done inside episode :', i_episode)
                    # print('ep_reward inside learn_policy_reset_random', ep_reward)
                    # print('----^ that should be >= 100 ^----')
                    # print('self.rewards in learn_policy_reset_random:', self.rewards)
                    state = self.init_state_reset_random(T)
                    #print('new initial state:', state)
                    #self.env.reset() #phil: change this

            running_reward = running_reward * 0.99 + ep_reward * 0.01
            if (i_episode == 0):
                running_reward = ep_reward
            # if finished:
            #     print('end of episode:', i_episode, ep_reward)
            #     print('end of episode:', i_episode, self.rewards)
            loss = self.update_policy()
            running_loss = running_loss * 0.99 + loss*.01

            # Log to console.
            if (i_episode) % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLoss: {:.2f}'.format(
                    i_episode, ep_reward, running_reward, running_loss))

        print('done training in learn_policy_reset random')

    def learn_policy(self, reward_fn, det_initial_state=False, sa_reward=True, true_reward=False,
        episodes=1000, train_steps=1000, 
        initial_state=[], start_steps=10000):

        if det_initial_state:
            initial_state = self.init_state
        # print("init: " + str(initial_state))

        running_reward = 0
        running_loss = 0
        for i_episode in range(episodes):
            #if i_episode % 5 == 0: #PHIL: changing this to every episode
            if det_initial_state:
                self.env.env.reset_state = initial_state
            self.env.reset()
            state = self.get_obs()
            #print('state in learn_policy at episode:', i_episode, state)
            # print('initial_state in learn_policy at episode:', i_episode, initial_state)
            ep_reward = 0
            for t in range(train_steps):  # Don't infinite loop while learning
                #print(state)
                action = self.select_action(state)
                if true_reward:
                    #prev_state = copy.deepcopy(state)
                    state, reward, done, _ = self.env.step(action)
                    #print('t, prev_state, action, state, reward in learn_policy:', t, prev_state, action, state, reward)     
                else:
                    #next_action = self.select_action(state)
                    tmp = copy.deepcopy(base_utils.discretize_state(state))
                    tmp.append(action[0])
                    if sa_reward:
                        reward = reward_fn[tuple(tmp)] #reward fn is a fn state-action pairs. applied before.
                        state, _, done, _ = self.env.step(action)
                    else:
                        state, _, done, _ = self.env.step(action)
                        reward = reward_fn[tuple(base_utils.discretize_state(state))] #reward fn is a fn of states. applied after.
                    del tmp
                ep_reward += reward
                self.rewards.append(reward)
                if done:
                    # TODO: self.env.env.reset_state = initial_state ? 
                    #print('done inside learn_policy')
                    #print('ep_reward inside learn_policy', ep_reward)
                    #print('----^ that should be >= 100 ^----')
                    #print('done inside learn_policy:')
                    if det_initial_state:
                        self.env.env.reset_state = initial_state
                    self.env.reset()
                    state = self.get_obs()
                    #print('resetting with state:', state)

            running_reward = running_reward * (i_episode)/float(i_episode+1) + ep_reward/float(i_episode+1)
            if (i_episode == 0):
                running_reward = ep_reward
            
            loss = self.update_policy()
            running_loss = running_loss * (i_episode)/float(i_episode+1) + loss/float(i_episode+1)

            gc.collect()

            # Log to console.
            if (i_episode) % 100 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLoss: {:.2f}'.format(
                    i_episode, ep_reward, running_reward, running_loss))

    #phil: modify this to also compute J(pi) for a given reward fn
    #phil: assuming the reward fn is a (deterministic) function of s,a
    def execute_internal(self, env, T, reward_fn, sa_reward, true_reward, state, render):
        # print("Simulation starting at = " + str(state))
        p = np.zeros(shape=(tuple(base_utils.num_states)))
        p_sa = np.zeros(shape=(tuple(base_utils.num_sa)))
        total_reward = 0.
        exited = False

        # print('starting execute_internal with state:', self.get_obs())
        # print('should be starting simulation at state:', str(state))
        #phil: this seems like its estimating for one trajectory only (or T steps, whichever comes first)
        #phil: can put an outer loop in execute
        for t in range(T):  
            p[tuple(base_utils.discretize_state(state))] += 1    
            action = self.select_action_no_grad(state)[0]
            tmp = copy.deepcopy(base_utils.discretize_state(state))
            tmp.append(action)
            p_sa[tuple(tmp)] +=1 
            if true_reward:
                state, reward, done, _ = env.step([action]) 
            else:
                if sa_reward:
                    reward = reward_fn[tuple(tmp)] #reward fn is a fn state-action pairs. applied before.
                    state, _, done, _ = env.step([action]) 
                else:
                    state, _, done, _ = env.step([action]) 
                    reward = reward_fn[tuple(base_utils.discretize_state(state))]  #reward fn is a fn of states. applied after update. 
            del tmp
            total_reward += reward
            
            #print('state, done, iter in execute_internal:', state, done, t)
            
            if render:
                env.render()
            if done:
                # print('done inside execute_internal')
                # print('total_reward inside execute_internal', total_reward)
                # print('----^ that should be 100 ^----')
                exited = True
                final_t = t + 1
                # print('breaking in execute_internal after t steps:', t)
                # print('total_reward in execute_internal:', total_reward)
                break
                #phil: can replace this with self.env.reset()?

        env.close()
        # print('p sums to: ', np.sum(p))
        # print('final_t:', final_t)
        # print('p/final_t is normalized:', np.sum(p/final_t))
        if exited:
            return p/float(final_t), p_sa/float(final_t), total_reward
        else:
            #print('did not exit in execute_internal')
            return p/float(T), p_sa/float(T), total_reward

    #phil: modify this to do several rollouts
    #phil: modify this to measure total reward
    def execute(self, T, reward_fn, sa_reward=True,true_reward=False,num_rollouts=1, initial_state=[], render=False, video_dir=''):

        p = np.zeros(shape=(tuple(base_utils.num_states)))
        p_sa = np.zeros(shape=(tuple(base_utils.num_sa)))
        total_reward = 0.

        for r in range(num_rollouts):
            #print('rollout number in execute:', r)
            if len(initial_state) == 0:
                initial_state = self.env.reset() # get random starting location

            # print("initial_state= " + str(initial_state))

            if render:
                print("rendering env in execute()")
                print("WARNING THIS IS NOT IMPLEMENTED CORRECTLY")
                wrapped_env = wrappers.Monitor(self.env, video_dir)
                wrapped_env.unwrapped.reset_state = initial_state
                state = wrapped_env.reset()
                state = self.get_obs()
                # print(initial_state)
                # print(state)
                p_int, _, total_reward_int = self.execute_internal(wrapped_env, T, reward_fn, state, render)
                p += p_int
                total_reward += total_reward_int
            else:
                self.env.env.reset_state = initial_state 
                state = self.env.reset()
                state = self.get_obs()

                # print('state inside execute:', state)
                # print('initial_state inside execute:', initial_state)
                p_int,p_sa_int,total_reward_int = self.execute_internal(self.env, T, reward_fn,sa_reward,true_reward,state,render)
                p += p_int
                p_sa += p_sa_int
                total_reward += total_reward_int

        # print('p sums to:', np.sum(p))
        # print('p_sa sums to:', np.sum(p_sa))
        # print('num_rollouts:', num_rollouts)
        # print('p/float(num_rollouts) sums to:', np.sum(p/float(num_rollouts)))
        # print('p_sa/float(num_rollouts) sums to:', np.sum(p_sa/float(num_rollouts)))
        return p/float(num_rollouts), p_sa/float(num_rollouts), total_reward/float(num_rollouts) #phil, return p_sa/float(num_rollouts) here 

    def execute_random_internal(self, env, T, reward_fn, state, render):
        p = np.zeros(shape=(tuple(base_utils.num_states)))
        p_sa = np.zeros(shape=(tuple(base_utils.num_sa)))
        total_reward = 0.
        exited = False

        # print('starting execute_random_internal with state:', self.get_obs())
        # print('should be starting simulation at state:', str(state))

        for t in range(T):  
            #can change this
            p[tuple(base_utils.discretize_state(state))] += 1
            r = random.random()
            action = -1
            if (r < 1/3.):
                action = 0
            elif r < 2/3.:
                action = 1

            tmp = copy.deepcopy(base_utils.discretize_state(state))
            tmp.append(action)
            reward = reward_fn[tuple(tmp)] #reward fn is a fn state-action pairs
            total_reward += reward
            p_sa[tuple(tmp)] +=1 
            del tmp
            state, _, done, _ = env.step([action])
            
            if render:
                env.render()
            if done:
                exited = True
                final_t = t + 1
                # print('breaking in execute_internal after t steps:', t)
                # print('total_reward in execute_internal:', total_reward)
                break
                #phil: can replace this with self.env.reset()?
            
        env.close()
        if exited:
            return p/float(final_t), p_sa/float(final_t), total_reward
        else:
            #print('did not exit in execute_random_internal') #just for debugging, this doesn't show up if T = the horizon
            return p/float(T), p_sa/float(T), total_reward

    #phil: modified this to take reward_fn as input (for measuring base l1 coverability)
    def execute_random(self, T, reward_fn, num_rollouts=1, initial_state=[], render=False, video_dir=''):
        p = np.zeros(shape=(tuple(base_utils.num_states)))
        p_sa = np.zeros(shape=(tuple(base_utils.num_sa)))
        total_reward = 0.

        for r in range(num_rollouts):
            if len(initial_state) == 0:
                #initial_state = self.env.reset() # get random starting location
                initial_state = self.init_state

            #print("initial_state= " + str(initial_state))

            if render:
                print("rendering env in execute_random()")
                print("RENDER IS NOT IMPLEMENTED CORRECTLY")
                wrapped_env = wrappers.Monitor(self.env, video_dir)
                wrapped_env.unwrapped.reset_state = initial_state
                state = wrapped_env.reset()
                state = self.get_obs()
                p = self.execute_random_internal(wrapped_env, T, state, render)
            else:
                self.env.env.reset_state = initial_state #phil: double-check what state we get reset to
                state = self.env.reset()
                state = self.get_obs()

                # print('state inside execute_random:', state)
                # print('initial_state inside execute_random:', initial_state)
                p_int, p_sa_int, total_reward_int = self.execute_random_internal(self.env, T, reward_fn, state, render)
                p += p_int
                p_sa += p_sa_int
                total_reward += total_reward_int

        return p/float(num_rollouts), p_sa/float(num_rollouts), total_reward/float(num_rollouts)

    def save(self, filename):
        self.env.close()
        torch.save(self, filename)