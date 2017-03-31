#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:08:39 2017

@author: yuedong
"""

import os
os.chdir("/Users/yuedong/Downloads/comp767_easy21/")
#%%

from easy21game import Easy21
import numpy as np
from matplotlib import cm
#%%
env = Easy21()

#%% code the state-action paris into features

# INPUT
#   playerState: sum of the player, integer between 1 and 21
# OUTPUT
#   boolean vector coding the player card interval on 6 bits
def playerFeatures(playerState):
    playerVec = np.array([playerState in [1,2,3,4,5,6], 
                     playerState in [4,5,6,7,8,9],
                     playerState in [7,8,9,10,11,12],
                     playerState in [10,11,12,13,14,15],
                     playerState in [13,14,15,16,17,18],
                     playerState in [16,17,18,19,20,21]]).astype('int')
    return np.where(playerVec)[0]

# INPUT
#   dealerState: card value of the dealer, integer between 1 and 10
# OUTPUT
#   boolean vector coding the dealer card to 3 intervals
def dealerFeatures(dealerState):
    dealerVec = np.array([dealerState in [1,2,3,4], dealerState in [4,5,6,7], 
                   dealerState in [7,8,9,10]]).astype('int')
    return np.where(dealerVec)[0]
  

# INPUT
#   action=0, stick    action=1, hit
# OUTPUT
#   boolean vector coding the player's action 
def actionFeatures(action):
    return np.array([action==0,action==1]).astype('int')


# INPUTS
#   s: state =(playerState,dealerState) (as defined in env._step)
#   a: action, integer: HIT(1) or STICK(0)
# returns a binary vector of length 36 representing the features
def phi(s, a):
    tmp = np.zeros(shape=(6,3,2)) #zeros array of dim 6*3*2
    #putting one where a feature is on
    for i in playerFeatures(s[0]):
        for j in dealerFeatures(s[1]):
            tmp[i,j,action] = 1 
    return(tmp.flatten()) #returning 'vectorized' (1-dim) array

#%% define the epsilon greedy function
# INPUTS
#   s: state 
#   Q: action-value function, that is an array of dim: (21, 10, 2)
#   eps: numeric value for epsilon
# OUTPUT
# action to take following an epsilong-greedy policy
# 1 is HIT and 2 is STICK (as defined in q1-step.R)
epsgreedy <- function(s, Q, eps) {
  if(runif(1)<eps)
    return(sample(1:2, 1)) # random action taken with eps probability
  else
    return(which.max(c(Q(s,HIT), Q(s,STICK)))) # else action maximizing Q
}
#%%
def sarsa_coarse_coding(LAMBDA, GAMMA=1, w=NULL, 
                        nb_episode=1, EPSILON=0.05, step_size=0.01):
  # setting w and N to their default value if necessary
  if (is.null(w))
    w <- np.zeros(36)

  # Q is simply the dot product of phi and w
  Q = np.dot(phi(s,a),w)

  # defining our policy (reference to Q is not closure)
  policy <- function(s) {
    epsgreedy(s, Q, eps)
  }

  for(i in 1:nb.episode) {
    s <- s.ini()
    # choosing initial action a
    a <- policy(s)
    r <- 0L
    # eligibility trace
    e <- array(0L, dim=36)

    # s[3] is the "terminal state" flag
    # s[3]==1 means the game is over
    while(s[3]==0L) {

      # performing step
      tmp <- step(s, a)
      s2 <- tmp[[1]]
      r <- r + tmp[[2]]

      # if s2 is terminal
      if (s2[3]==0) {
        # choosing new action
        a2 <- policy(s2)
        # sarsa backward view formula with estimated return r+gamma*Q2
        delta <- r + gamma*Q(s2,a2) - Q(s,a)
      } else {
        # sarsa backward view formula, with now known return r
        delta <- r - Q(s,a)
        a2 <- 0L
      }
      ind <- which(e>0)
      # updating eligibility traces and w
      e <- gamma*lambda*e + phi(s,a)
      w <- w + step.size*delta*e
      s <- s2
      a <- a2
    }
  }
  return(w)
}

#%%
class Sarsa_Agent_LA:
    def __init__(self, environment, mlambda):
        self.env = environment
        self.mlambda = mlambda
        
        self.iterations = 0
    

          # get optimal action based on ε-greedy exploration strategy  
    def fixed_epsilon_greedy_action(self, state, epsilon=0.1):
        player_idx, dealer_idx = self.get_state_number(state)
        # action = 0 stick
        # action = 1 hit
        hit = 1
        stick = 0
        # epsilon greedy policy
        if np.random.random() < epsilon:
            r_action = hit if np.random.random()<0.5 else stick
            return r_action
        else:
            action = np.argmax(self.Q[dealer_idx, player_idx, :])
            return action
      
#              # get optimal action based on ε-greedy exploration strategy
#              # the epsilon varies based on the number of visits
#    def var_epsilon_greedy_action(self, state, epsilon=0.1):
#        dealer_idx = state[0]
#        player_idx = state[1]
#        # action = 0 stick
#        # action = 1 hit
#        hit = 1
#        stick = 0
#        # epsilon greedy policy
#        if np.random.random() < epsilon:
#            r_action = hit if np.random.random()<0.5 else stick
#            return r_action
#        else:
#            action = np.argmax(self.Q[dealer_idx, player_idx, :])
#            return action
        
    def get_action(self, state):
        player_idx, dealer_idx = self.get_state_number(state)
        action = np.argmax(self.Q[dealer_idx, player_idx, :])
        return action
    
    
    
#    def validate(self, iterations):        
#        wins = 0; 
#        # Loop episodes
#        for episode in xrange(iterations):
#
#            s = self.env.get_start_state()
#            
#            while not s.term:
#                # execute action
#                a = self.get_action(s)
#                s, r = self.env.step(s, a)
#            wins = wins+1 if r==1 else wins 
#
#        win_percentage = float(wins)/iterations*100
#        return win_percentage
    
    def train(self, iterations):        
        
        # Loop episodes
        for episode in range(iterations):
            self.E = np.zeros((self.env.dealer_space,
                               self.env.player_space, self.env.nA))

            # get initial state for current episode
            s = self.env._reset()
            a = self.fixed_epsilon_greedy_action(s)
            a_next = a
            term = False
            
            # Execute until game ends
            while not term:
                # update visits
                player_idx, dealer_idx = self.get_state_number(s)
                self.N[dealer_idx, player_idx, a] += 1
                
                # execute action
                s_next, r, term = self.env._step(a)[0:3]
                player_idx_next, dealer_idx_next = self.get_state_number(s_next)
                q = self.Q[dealer_idx, player_idx, a]
                                
                if not term:
                    # choose next action with epsilon greedy policy
                    a_next = self.fixed_epsilon_greedy_action(s_next)
                    next_q = self.Q[dealer_idx_next, player_idx_next, a_next]
                    delta = r + next_q - q
                else:
                    delta = r - q
                
#                 alpha = 1.0  / (self.N[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)])
#                 update = alpha * delta
#                 self.Q[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)] += update
                
                self.E[dealer_idx, player_idx, a] += 1
                alpha = 1.0  / (self.N[dealer_idx, player_idx, a])
                update = alpha * delta * self.E
                self.Q += update
                self.E *= self.mlambda

                # reassign s and a
                s = s_next
                a = a_next

            #if episode%10000==0: print "Episode: %d, Reward: %d" %(episode, my_state.rew)
            self.count_wins = self.count_wins+1 if r==1 else self.count_wins

        self.iterations += iterations
#       print float(self.count_wins)/self.iterations*100

        # Derive value function
        for d in range(self.env.dealer_space):
            for p in range(self.env.player_space):
                self.V[d,p] = max(self.Q[d, p, :])
                
    def plot_frame(self, ax):
        def get_stat_val(x, y):
            return self.V[x, y]

        X = np.arange(0, self.env.dealer_space, 1)
        Y = np.arange(0, self.env.player_space, 1)
        X, Y = np.meshgrid(X, Y)
        Z = get_stat_val(X, Y)
        #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        return surf
    
#%%

N0 = 100
agent = Sarsa_Agent(env, N0, 0.9)

for i in range (100):
    agent.train(1000)

agent.V

##%%
#
#N0 = 100
#lambdas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#agent_list = []
#sme_list = []
##n_elements = mc_agent.Q.shape[0]*mc_agent.Q.shape[1]*2
#for l in lambdas:
#    agent = Sarsa_Agent(Environment(), N0, l)
#    agent_list.append(l)
#
#    agent.train(1000)
#    #sme = np.sum(np.square(agent.Q-mc_agent.Q))/float(n_elements)
#    #sme_list.append(sme)

#%%

def animate(frame):
    i = agent.iterations
    step_size = i
    step_size = max(1, step_size)
    step_size = min(step_size, 2 ** 16)
    agent.train(step_size)

    ax.clear()
    surf =  agent.plot_frame(ax)
    plt.title('MC score:%s frame:%s step_size:%s ' % (float(agent.count_wins)/agent.iterations*100, frame, step_size) )
    # plt.draw()
    fig.canvas.draw()
    print("done ", frame, step_size, i)
    return surf

#%%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
%matplotlib inline 

N0 = 100
mlambda = 0.2
agent = Sarsa_Agent(env, N0, mlambda)
fig = plt.figure("N100")
ax = fig.add_subplot(111, projection='3d')

# ani = animation.FuncAnimation(fig, animate, 32, repeat=False)
ani = animation.FuncAnimation(fig, animate, 10, repeat=False)

# note: requires gif writer; swap with plt.show()
ani.save('Sarsa_Agent_py.gif', writer='imagemagick', fps=3)
# plt.show()