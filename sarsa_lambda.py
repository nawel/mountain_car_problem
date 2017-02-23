%matplotlib inline

import gym
import itertools
import matplotlib
import numpy as np
import tiles
env = gym.envs.make("MountainCar-v0")


class Sarsa_lambda():
    """
    Control method with function approximation
    """
    
    def __init__(self, nteta=300, gamma=1, lmbda=0.9, alpha=0.9, epsilon=0.05):
       
        self.gamma=gamma
        self.lmbda=lmbda
        self.alpha=alpha
        self.epsilon=epsilon
        self.nteta=nteta #number of parameters
        
        
        self.ntiles=10
        
        self.Q=np.zeros(3) #action value function
        self.teta=np.zeros(nteta) #parameter of gradient descent
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
    
    def learn(self, n_episodes, maxstep):
        " Run SARSA lambda algorithm over n_episodes"
        
        stats = plotting.EpisodeStats(episode_lengths=np.zeros(n_episodes),episode_rewards=np.zeros(n_episodes)) 
    
        for e in range(n_episodes):
            # init the environement and take the first state
            state = env.reset()
                       
            #clear eligibility traces
            self.z=np.zeros(self.nteta)
            
            #compute features for this state
            self.compute_features(state)
            
            #compute new states values for action value
            self.compute_action_value(None)
            
            #take the argmax with epsilon probability
            action=np.argmax(self.Q)
            if np.random.rand() < self.epsilon:
                action= np.random.choice(3,1)[0]
            
            for stp in range(maxstep):
                #loop over steps in same episode
                
                for f in range(self.ntiles):
                    for a in range(3):
                        if (a!=action):
                            self.z[self.features[a][f]]=0
                    
                for f in range(self.ntiles):
                    self.z[self.features[action][f]]=1
                    
                state, reward, done, _ = env.step(action)
                
                stats.episode_rewards[e] += reward
                stats.episode_lengths[e] = stp
                
                

                delta=reward-self.Q[action]

                if not done:
                    #compute features for new  state
                    self.compute_features(state)

                    #compute new states values for action value
                    self.compute_action_value(None)

                    action=np.argmax(self.Q)

                    #epsilon greedy policy
                    if np.random.rand() < self.epsilon:
                        action= np.random.choice(3,1)[0]

                    delta += self.gamma * self.Q[action]

                self.teta+= (self.alpha/self.ntiles)*delta *self.z
                self.compute_action_value(action)
                self.z*=self.gamma*self.lmbda #decaying traces
        
        return stats

        
