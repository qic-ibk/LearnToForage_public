# -*- coding: utf-8 -*-
"""
Copyright 2020 Andrea López Incera.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Andrea López Incera, used and analysed in, 

'Development of swarm behavior in artificial learning agents that adapt to different foraging environments.'
Andrea López-Incera, Katja Ried, Thomas Müller and Hans J. Briegel.

"""

import numpy as np
import copy

class NutritionalAdaptation_nosmell(object):
    
    
    def __init__(self, num_agents, world_size, sensory_range, blind, distance_food, steps_randomizer, env_type, init_region, mode_samepos):
        """Initializes a world. Arguments:
        num_agents (int>0) - number of agents
        world_size (int>0) - length of world; ends are identified (ie world is circular)
        sensory range (int>0) - how many steps away at the front OR back an agent can see others.
        blind - if True, agents cannot see anything.
        distance_food - distance from the center of initialization region at which food source is placed.
        steps_randomizer - every steps_randomizer interaction rounds, the orientation of the agent is randomized.
        env_type- 'scarcity'(little food concentrated in small area) or 'flourish'(lot of food around the world).
        init_region - number of positions that constitute the initialization region, where agents are placed at the beginning of the trial.
        mode_samepos - 'random' (with 50% prob. focal agent sees agents placed at its same position to be behind. with 50% to be in front) or 'split' (focal agent sees half of the agents placed at its same position in front, and half behind)."""
        
        self.num_agents = num_agents;
        self.world_size = world_size;
        self.sensory_range = sensory_range;
        self.env_type=env_type;
        self.distance_food=distance_food;
        self.steps_randomizer=steps_randomizer;
        self.blind=blind; #this can be True or False. 
        self.init_region=init_region;#region (number of positions from position 0) where agents are initialized.
        self.mode_samepos=mode_samepos;#what to do if self sees several agents in the same position as self. 
        
        self.num_actions=2 #turn (0), go  (1) .
        self.num_percepts_list = [5,5] #how agents are walking in front and behind.
       
     
        self.positions = np.random.choice(int(self.init_region),self.num_agents) 
        
        self.speeds=np.random.choice([-1,1],self.num_agents)#orientatations of agents.
        self.randomizer=np.random.choice(10,self.num_agents)
        
        
        self.allagents_neighbours=[0]*self.num_agents

        if self.env_type=='flourish':#one half of the world has food, evenly distributed.
            self.food_distribution=np.ndarray.tolist(np.random.choice(self.world_size,int(0.5*self.world_size)))
            #in which positions of the world there is food.
        if self.env_type=='scarcity':#food is situated far away from the point agents can reach with a random walk.
            self.food_distribution=[self.distance_food+0.5*self.init_region,(0.5*self.init_region-self.distance_food)%self.world_size]
            #in which positions of the world the food is situated.
            
        
     
    def get_neighbours(self,agent_index):
        """Determine indices of all agents within visual range excluding self."""
        focal_pos = self.positions[agent_index];
        if self.blind:
            neighbours=[]
        else:
            neighbours = np.ndarray.tolist(np.where(dist_mod(self.positions,focal_pos,self.world_size)<self.sensory_range+1)[0]);
            neighbours.remove(agent_index)
      
        self.allagents_neighbours[agent_index]=neighbours
        return()
        
    def net_rel_mvmt(self,agent_index):
        """Returns an array with the number of agents that are receding or approaching from/to self in the regions back and front. The array
        has the form [0,#rec,#app]."""
        
        rel_walking_b=np.zeros(3)
        rel_walking_f=np.zeros(3)
        
        truth_back=((self.positions[agent_index]-self.positions[np.asarray(self.allagents_neighbours[agent_index])])%self.world_size<=self.sensory_range)*((self.positions[agent_index]-self.positions[np.asarray(self.allagents_neighbours[agent_index])])%self.world_size!=0)#if agents are at the back, the value at the corresponding agent index is 1 (true).        
        rel_dir_back=self.speeds[np.asarray(self.allagents_neighbours[agent_index])][np.where(truth_back==1)]#selects the speeds of locusts at the back.
        rel_walking_b[2]=np.sum(np.sign(rel_dir_back+1))#how many are approaching
        rel_walking_b[1]=np.sum(np.sign(-1*(rel_dir_back)+1))#how many are receding
        
        truth_front=(self.positions[agent_index]-self.positions[np.asarray(self.allagents_neighbours[agent_index])])%self.world_size>self.sensory_range
        rel_dir_front=self.speeds[np.asarray(self.allagents_neighbours[agent_index])][np.where(truth_front==1)]
        rel_walking_f[2]=np.sum(np.sign(-1*(rel_dir_front)+1))#approaching
        rel_walking_f[1]=np.sum(np.sign(rel_dir_front+1))#receding
        
        truth_same=((self.positions[agent_index]-self.positions[np.asarray(self.allagents_neighbours[agent_index])])%self.world_size<=self.sensory_range)*((self.positions[agent_index]-self.positions[np.asarray(self.allagents_neighbours[agent_index])])%self.world_size==0)#if agents are in the same position as self, the value at the corresponding agent index is 1 (true).
        rel_dir_same=self.speeds[np.asarray(self.allagents_neighbours[agent_index])][np.where(truth_same==1)]#selects the speeds of agents at the same position as self.

        if self.mode_samepos=='random':
            if np.random.choice(2):#with 50% prob., the agent sees them in the front.
                rel_walking_f[2]+=np.sum(np.sign(-1*(rel_dir_same)+1))#approaching
                rel_walking_f[1]+=np.sum(np.sign(rel_dir_same+1))#receding
            else:#with 50% prob., the agent sees them in the back.
                rel_walking_b[2]+=np.sum(np.sign(rel_dir_same+1))#how many are approaching
                rel_walking_b[1]+=np.sum(np.sign(-1*(rel_dir_same)+1))#how many are receding
                
        if self.mode_samepos=='split':
            rel_walking_f[2]+=np.sum(np.sign(-1*(rel_dir_same)+1))/2#approaching
            rel_walking_f[1]+=np.sum(np.sign(rel_dir_same+1))/2#receding
            rel_walking_b[2]+=np.sum(np.sign(rel_dir_same+1))/2#how many are approaching
            rel_walking_b[1]+=np.sum(np.sign(-1*(rel_dir_same)+1))/2#how many are receding
            
        if self.speeds[agent_index]==1:
            return(rel_walking_b,rel_walking_f)
        else:
            return(rel_walking_f,rel_walking_b)
        
        
    def where_max(self,relative_walking):
        """Given an array with the number of agents that walk in each different way with respect to self, 
        it computes how the majority of them walk. And if there is a high or small density in that region. 
        0 nobody, 1 receding small dens, 2 receding high dens, 3 approaching small den,4 approaching high den."""
        way=np.where(relative_walking==max(relative_walking))[0]
        if np.size(way)>1:
            if way[0]==0:
                return 0#nobody in that region
            if way[0]!=0:#if the same number of agents are walking in several directions, it will choose one randomly.
                fa=np.random.choice(way)
                if np.sum(relative_walking)>2:#there are more than 2 agents in the region.
                    return 2*fa
                else:
                    return 2*fa-1

        else:
            if np.sum(relative_walking)>2: #there are more than 2 agents in the region.
                return 2*way[0]
            else:
                return 2*way[0]-1
                
    
    def get_percept(self,agent_index):
        """Given an agent index, returns the percept [how majority of agents in front walk,how maj.of agents at the back walk], with
        0 nobody, 1 receding small density,2 rec big den, 3 approaching small density, 4 app big den."""
        if len(self.allagents_neighbours[agent_index])==0:#self has no neighbours
            return([0,0])
        else:
            rel_walking_b,rel_walking_f=self.net_rel_mvmt(agent_index)
            return([self.where_max(rel_walking_f),self.where_max(rel_walking_b)])
        
    def move(self,agent_index, action, test_reward):
        """Given an agent index and that agent's action (0 for keep going, 1 for turning), this function updates everything and computes its
        reward, along with the percept for the next agent in the list."""
        reward=0
        if action==1:#turn 
            self.speeds[agent_index]=-1*self.speeds[agent_index]
        
            
        #update of positions and randomizer:
        self.positions[agent_index]=np.remainder(self.positions[agent_index]+self.speeds[agent_index],self.world_size)
        self.randomizer[agent_index]+=1 
         
        #process of eating: 
        if test_reward==0 and self.positions[agent_index] in self.food_distribution:
            reward=1
        
        #get info for next agent
        self.get_neighbours((agent_index+1)%self.num_agents)#update the list of neighbours of the next agent, because it may have changed due to the movement of self.
                    
        if self.randomizer[(agent_index+1)%self.num_agents]%self.steps_randomizer==0:#randomize direction of next agent every steps_randomizer steps.
            self.speeds[(agent_index+1)%self.num_agents]=np.random.choice([1,-1])
        
        next_percept=self.get_percept((agent_index+1)%self.num_agents)
        return(next_percept,reward)
    
    def reset(self):
        """Sets positions and speeds back to random values and returns the percept for the
        0th agent. """
        
        self.positions = np.random.choice(int(self.init_region),self.num_agents) 
        
        self.speeds=np.random.choice([-1,1],self.num_agents)
        self.get_neighbours(0)

        return(self.get_percept(0))
        
    
        
def dist_mod(num1,num2,mod):
    """Distance between num1 and num2 (absolute value)
    if they are given modulo an integer mod, ie between zero and mod.
    Also works if num1 is an array (not a list) and num2 a number or vice versa."""
    diff=np.remainder(num1-num2,mod)
    diff=np.minimum(diff, mod-diff)
    return(diff)