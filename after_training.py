# -*- coding: utf-8 -*-
"""
Copyright 2020 Andrea López Incera.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Andrea López Incera, used and analysed in, 

'Development of swarm behavior in artificial learning agents that adapt to different foraging environments.'
Andrea López-Incera, Katja Ried, Thomas Müller and Hans J. Briegel.

This piece of code is used in all the analyses where agents are already trained, and we study features of the resulting dynamics.
For instance, it is used to generate the trajectories analyzed in terms of foraging models (BW, PL, CRW or CCRW).

"""
import numpy as np
import numpy.ma as ma
import copy 
import pickle


import environment
import ps_agent_foraging
import time

#parameters we set:
num_agents=60
world_size=500#world size should be much bigger than sensory range.
sensory_range=6
init_region=2*sensory_range
mode_samepos='split'
blind=False
distance_food=21
steps_randomizer=5
env_type='scarcity'
gamma_damping=0#float between 0 and 1. Forgetting.
eta_glow_damping=0.2#float between 0 and 1. Setting it to 1 effectively deactivates glow.
policy_type='standard'#usual computation of prob distribution according to h matrix.
beta_softmax=1#irrelevant if policy_type is standard.
num_reflections=0 #effectively deactivates reflections.
del_steps=100000 #number of deliberation steps per trial.
num_trials=1
num_pop=10

#load already trained populations

agent_list=[0]*num_pop
for population in range(num_pop):
    agent_list[population],which_trial=pickle.load(open( 'save_agents_d21_pop={0}'.format(population)+'.txt', "rb" ))
    

#initialization of the objects "environment" .
env=environment.NutritionalAdaptation_nosmell(num_agents, world_size, sensory_range, blind, distance_food, steps_randomizer, env_type,init_region,mode_samepos)


#initialize a record of performance
alignment=np.zeros([num_pop,del_steps])
local_alignment=np.zeros([num_pop,del_steps])
cohesion=np.zeros([num_pop,del_steps])


record_pos_end=[0]*num_pop
for p in range(num_pop):
    record_pos_end[p]=np.zeros([del_steps,num_agents])



#interaction


for population in range(num_pop):
    reward_trial_list=np.zeros(num_agents)#reset reward list and test of rewards
    reward_list=np.zeros(num_agents)
    test_reward=np.zeros(num_agents)
    
  
    next_observation=env.reset()#reset positions of all agents and take percept for first agent.
    
    
    for t in range(del_steps):
        record_pos_end[population][t]=env.positions
        counter_align=0
        local_align=0
        cohes=0
        for i in range(num_agents):
            action=agent_list[population%20][i].deliberate_and_learn(next_observation,reward_list[i])
            next_observation,reward_list[i]=env.move(i,action,test_reward[i])
#            if reward_list[i]==1: #agents do not learn anymore in this simulation.
#                test_reward[i]=1
                  
            reward_trial_list[i]=test_reward[i]
            
            cohes+=len(env.allagents_neighbours[i])#number of neighbours inside sensory range of agent i.
            counter_align+=env.speeds[i]  
            if len(env.allagents_neighbours[i])==0:
                local_align+=0
            else:
                local_align+=abs(np.sum(env.speeds[np.asarray(env.allagents_neighbours[i])])/len(env.allagents_neighbours[i]))
    
        alignment[population,t]=counter_align/num_agents
        local_alignment[population,t]=local_align/num_agents
        cohesion[population,t]=cohes/num_agents
        
        

#np.savetxt('local_alignment_levy_d21.txt',local_alignment)
#np.savetxt('alignment_levy_d21.txt',alignment)
#np.savetxt('cohesion_study_d21.txt',cohesion)
#file = open( "positions_traj_levy_d21.txt", "wb" )
#pickle.dump(record_pos_end,file)
#file.close()

