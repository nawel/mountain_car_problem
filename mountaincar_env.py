"""
Simulation of mountain car environement
"""
import numpy as np

def reset_env():
    position = -0.6 + np.random.rand()*0.2
    return [position, 0.0]

def next_step(state,action):
    """
    Move to next state
    """
    [position,velocity] = state
    if not action in (0,1,2):
        print 'Invalid action:', action
        raise StandardError
    R = -1
    velocity += 0.001*(action-1) - 0.0025*np.cos(3*position)
    if velocity < -0.07:
        velocity = -0.07
    elif velocity >= 0.07:
        velocity = 0.07
    position += velocity
    if position >= 0.5:
        return R,None,True
    if position < -1.2:
        position = -1.2
        velocity = 0.0
    return R,[position,velocity],False
