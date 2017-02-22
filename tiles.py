# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:37:59 2017

@author: GEOL05
"""

import numpy as np
import math
import random

MAX_NUM_VARS=20   # indicates the maximum number of vars in a tile grid
rndSeq = [random.randrange(2127483647//4) for i in range(2048)]  # a sequence table of random integers used for hashing
Qstate = np.zeros(MAX_NUM_VARS)
base = np.zeros(MAX_NUM_VARS)

#function used to hash the array of coordinates into tiles
def hashCoordinates(coord, nrCoord, memSize):
    
    Sum = 0
    inc = 449
   
    for i in range(nrCoord):
        Sum +=rndSeq[(int(coord[i]) + i*inc) %2048]
    
    return Sum % memSize  

#this function takes as input the number of tile indices, an array of real number variables
#the total number of possible tiles and an array of integer variables (can be empty)
#returns an array of tile indices coresponding to the variables
def getTiles(nrTileInd, var, memSize, intVars=[]):
    
   global Qstate, base
   nrVars = len(var) 
   nrCoord = nrVars +1 +len(intVars)
   coord = np.zeros(nrCoord)
   tiles = np.zeros(nrTileInd)
   
   i=nrVars +1
   for var in intVars:
       coord[i] = var
       i+=1
         
   for i in range(nrVars):
       Qstate[i]= int(math.floor(var[i]*nrTileInd))
       base[i] = 0;
       
   for j in range(nrTileInd):
       for i in range(nrVars):
           if Qstate[i] >= base[i]:
               coord[i] = Qstate[i] - ((Qstate[i] - base[i]) % nrTileInd)
           else:
               coord[i] = Qstate[i] + 1+ ((base[i] - Qstate[i] -1) % nrTileInd) - nrTileInd
               
           base[i] += 1 + (2*i)
       coord[nrVars] = j;
       tiles[j] = hashCoordinates(coord, nrCoord, memSize)

   return tiles

#similar to getTiles, but used to load nrTileInd tiles into an existing tiles array, starting from position: start
def loadTiles(tiles, start, nrTileInd, var, memSize, intVars=[]):
   
   global Qstate, base
   nrVars = len(var) 
   nrCoord = nrVars +1 +len(intVars)
   coord = np.zeros(nrCoord)
   
   i=nrVars +1
   for var in intVars:
       coord[i] = var
       i+=1
         
   for i in range(nrVars):
       Qstate[i]= int(math.floor(var[i]*nrTileInd))
       base[i] = 0;
       
   for j in range(nrTileInd):
       for i in range(nrVars):
           if Qstate[i] >= base[i]:
               coord[i] = Qstate[i] - ((Qstate[i] - base[i]) % nrTileInd)
           else:
               coord[i] = Qstate[i] + 1+ ((base[i] - Qstate[i] -1) % nrTileInd) - nrTileInd
               
           base[i] += 1 + (2*i)
       coord[nrVars] = j;
       tiles[start+ j] = hashCoordinates(coord, nrCoord, memSize)
    
        
tiles = getTiles(1, [2.0, 2.0], 4096)
print (tiles)
    
