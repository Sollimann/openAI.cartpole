# Written by Kristoffer Rakstad Solberg
# For Assignment 1 in TTK23

# coding: utf-8

# In[1]:


import gym
from gym import wrappers
import math as m
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')


# In[2]:


# Q-learning parameters
MAXSTATES = 10000
ALPHA = 0.01 # [0,1] A very small learning rate, might take long to converge
GAMMA = 0.9 # [0,1] A large discount factor means that you value future actions much
MAX_POS = 2.5
MAX_VEL = 5
MAX_ANGLE = float(30 * m.pi/180)
MAX_TIP_VEL = 5


# In[3]:


# this function creates a sample space by using a linear function
# val = (MAX+MIN)/10 * x + MIN
def discrete_ss():
    ss = np.zeros((4,10)) #state space / sample space
    ss[0] = np.linspace(-MAX_POS,MAX_POS,10)
    ss[1] = np.linspace(-MAX_VEL,MAX_VEL,10)
    ss[2] = np.linspace(-MAX_ANGLE,MAX_ANGLE,10)
    ss[3] = np.linspace(-MAX_TIP_VEL,MAX_TIP_VEL,10)

    return ss


# In[4]:


# Problem: For a given state, what bin/discrete area does it fall into
# Return the indices of the bins/states to which each value in input array belongs
# observation is the input array to be binned
# ss is the state space / array of bins
def assign_discrete_state(observation,ss):
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i],ss[i]) #digitize turns the continous observation into discrete ss
    return state


# In[5]:


# This function takes a 4x1 num array and returns it as a 4x1 string array
def num_to_string(state):
    return ''.join(str(int(num)) for num in state)


# In[6]:


# Converting all four states into strings
def get_all_states_as_string():
    states = [] #creating an empty array
    for i in range(MAXSTATES):
        states.append(str(i).zfill(4)) #zfill will make sure that the size of the appended str is 4
    return states


# In[7]:


# Initializing the Q dictionary
def initialize_Q():
    Q = {}
    all_states = get_all_states_as_string() # converting all states to strings
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0 # For each action we assign an expected future reward to be zero. Value doesnt matter
    return Q


# In[8]:


# Iteration through a dictionary d finding states and associating actions
# This function will find the action that gives the highest expected reward
# This function could have been made general by iterating through possible actions in Qlearning dictionary
def max_dictionary(d):
    max_key = 0
    max_reward = d[0]
    if max_reward < d[1]:
        max_key = 1
        max_reward = d[1]
    return max_key, max_reward


# In[9]:


def Qlearning_agent(ss, Q, epsilon, ep):
    observation = env.reset()
    accumulated_reward = 0
    moves = 0
    done = False #flag
    state = num_to_string(assign_discrete_state(observation,ss)) #converting a observation to discrete string
    
    while not done:
        moves += 1 #adding up the moves
        
        if ep > 9998:
            env.render() #if we want to do RL we want to comment this out because it takes too much time
        
        # np.random.uniform() gives a float number between 0 and 1. If epsilon is a large number less than 1,
        # and larger than 0.5, then the algorithm will explore more random actions compared to picking actions
        # from it dictionary. With epislon 0.5 the agent will pick random actions more or less each other time
        if np.random.uniform()< epsilon:
            action = env.action_space.sample() # Testing random actions from action_space
        else:
            # action = [max_key][0]. Q[state] = Q[0,0,1,0] = 4x10 array
            # actions are picked from the discrete action space. Best action is the one with highest reward
            action = max_dictionary(Q[state])[0]
            
        # env.step executes the action and returns the observed state and reward. Also whether or not the system fails    
        observation, reward, done, info = env.step(action)
        
        # the environment returns +1 reward for each successfull action
        accumulated_reward += reward
        
        # If the agent performs less than 200 moves and fails
        # then punish the agent for pole falling over
        if done and moves < 200:
            reward = -300
        
        # The observed state of the environment is collected
        state_observed_after_act = num_to_string(assign_discrete_state(observation,ss))
        
        # Returning the highest possibly rewarded act for the observed new state
        # future act is either 0 or 1 depending on which act has the highest award
        future_act, max_Q_reward = max_dictionary(Q[state_observed_after_act])
        #print('State observed', state_observed_after_act)
        #print('Qvalue: ', Q[state_observed_after_act])
        #print('future_act: ',future_act, ', max_reward: ',max_Q_reward)
        
        # evaluating the last move updating the Qlearning dictionary
        Q[state][action] += ALPHA*(reward + GAMMA*max_Q_reward - Q[state][action])
        
        # The new state is saved for next run in the while loop
        state = state_observed_after_act
        
        #accumulated_reward and number of moves will be equal in this case
    return accumulated_reward, moves


# In[10]:


def Qlearning_training(ss,GAME=10000):
    Q = initialize_Q()
    length = []
    reward = []
    epsilons = []
    
    # Running through all episodes of the GAME
    for ep in range(GAME):
        
        # epsilon gradually decreases, thus trusting the learned actions more and more
        #epsilon = 1.0 / np.sqrt(ep+1)
        epsilon = 1.0/ np.power(ep+1,1/2)
        
        # episode statistics
        episode_reward, episode_length = Qlearning_agent(ss,Q,epsilon, ep)
        length.append(episode_length)
        reward.append(episode_reward)
        epsilons.append(epsilon)
        
        # printing
        if ep % 100 == 0:
            print(ep, '%.4f' % epsilon, episode_length)
            
    return length, reward, epsilons


# In[11]:


def plot_running_epsilons(epsilons):
    fig = plt.figure()
    plt.plot(epsilons)
    plt.xlabel('episode')
    plt.ylabel('epsilon')
    plt.title("Running epsilons")
    fig.savefig('epsilons.jpg')
    
def plot_running_avg(rewards):
    N = len(rewards)
    running_avg = np.empty(N)
    
    for n in range(N):
        running_avg[n] = np.mean(rewards[max(0,n-100):(n+1)])
    
    fig = plt.figure()
    plt.plot(running_avg)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title("Running Average, alpha=0.01")
    fig.savefig('rewards.jpg')


# In[12]:


if __name__ == '__main__':
    ss = discrete_ss()
    lengths, rewards, epsilons = Qlearning_training(ss)
    plot_running_avg(rewards)
    plot_running_epsilons(epsilons)

