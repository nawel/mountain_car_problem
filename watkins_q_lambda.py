%matplotlib inline
import random
import gym
import itertools
import matplotlib
import numpy as np
import tiles
env = gym.envs.make("MountainCar-v0")
import sys
if "../" not in sys.path:
    sys.path.append("../") 

from lib import plotting
matplotlib.style.use('ggplot')
#from mountaincar_env import *

class Watkins_Q_lambda():
    """
    Control method with function approximation
    """
    
    def __init__(self, nteta=3000, gamma=0.8, lmbda=0.1, alpha=0.05, epsilon=0.05):
       
        self.gamma=gamma
        self.lmbda=lmbda
        self.alpha=alpha
        self.epsilon=epsilon
        self.nteta=nteta #number of parameters
        
        
        self.ntiles=10
        
        self.Q=np.zeros(3) #action value function
        self.teta=theta = -0.01*np.random.randn(nteta) #parameter of gradient descent
        self.features=np.zeros((3,10)) #matrix of features (action, feature)
        self.z=np.zeros(nteta) #eligibility traces vector
        
        
    def compute_action_value(self, action):
        " Compute action value for current step"
        if action==None:
            self.Q=np.zeros(3)
            for a in range(3):
                for f in range(self.ntiles): 
                    self.Q[a]+=self.teta[self.features[a][f]]
        else:
            self.Q[action]=0
            for f in range(self.ntiles): 
                self.Q[action]+=self.teta[self.features[action][f]]
                
    def compute_features(self, state):
        "Compute features for currunt state"
        
        for a in range(3):
            self.features[a]=tiles.getTiles(self.ntiles, state, self.nteta,intVars=[a])
            
    def greedy(self):
        "Compute action with epsilon greedy"
        if np.random.rand() < self.epsilon:
            return np.random.choice(3,1)[0]
        else:
            return np.argmax(self.Q)            
           
    
    def learn(self, n_episodes, maxstep):
        " Run Watkin's Q(lambda) algorithm over n_episodes"
        
        stats = plotting.EpisodeStats(episode_lengths=np.zeros(n_episodes),episode_rewards=np.zeros(n_episodes)) 
    
        for e in range(n_episodes):
            
            state = reset_env() # init the environement and take the first state
            self.z=np.zeros(self.nteta)  #clear eligibility traces
            
            for stp in range(maxstep):#loop over steps in same episode
                
                self.compute_features(state) #compute features for this state
                self.compute_action_value(None)#compute new states values for action value
                action=self.greedy()  #take the argmax with epsilon probability

                reward, new_state, done= next_step(state,action)
                
                #stats
                stats.episode_rewards[e] += reward
                stats.episode_lengths[e] = stp
                
                delta=reward-self.Q[action]
                
                for f in range(self.ntiles):
                    for a in range(3):
                        if(a!=action):
                            self.z[self.features[a][f]]=0
                    self.z[self.features[action][f]]=1
                    
                if done:
                    self.teta+= (self.alpha/self.ntiles)*delta *self.z
                    break
                                  
                self.compute_features(new_state) #compute features for new  state
                self.compute_action_value(None) #compute new states values for action value
                
                q_max=np.max(self.Q)
                delta += self.gamma * q_max

                self.teta+= (self.alpha/self.ntiles)*delta *self.z
                #self.compute_action_value(action)    
                self.z=self.gamma*self.lmbda * self.z#decaying trace
                state=new_state
                
        return stats

        
