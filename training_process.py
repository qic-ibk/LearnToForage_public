# -*- coding: utf-8 -*-
"""
Copyright 2020 Andrea López Incera.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Andrea López Incera, used and analysed in, 

'Development of swarm behavior in artificial learning agents that adapt to different foraging environments.'
Andrea López-Incera, Katja Ried, Thomas Müller and Hans J. Briegel.

For agents defined in ps_agent_foraging.py and environment defined in environment.py, this piece of code performs the learning process
of 20 populations, given one value of dF (distance_food). In our analysis, we have considered different learning processes by changing dF. For each value of dF, this piece of code was used.

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
del_steps=50 #number of deliberation steps per trial.
num_trials=10000
num_pop=20

#initialization of the objects "environment" .
env=environment.NutritionalAdaptation_nosmell(num_agents, world_size, sensory_range, blind, distance_food, steps_randomizer, env_type,init_region,mode_samepos)

#record of performance.
alignment_pop=np.zeros([num_pop,num_trials])
local_alignment_pop=np.zeros([num_pop,num_trials])
cohesion_pop=np.zeros([num_pop,num_trials])
learning_pop=np.zeros([num_pop,num_trials])

for population in range(num_pop):
    #initialization of the objects "PSagents" .
    agent_list=[]
    for i in range(num_agents):
        agent_list.append(ps_agent_foraging.BasicPSAgent(env.num_actions,env.num_percepts_list,\
        gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections))
     
    
    #initialize a record of performance for one population.
    alignment=np.zeros([num_trials,del_steps])
    learning_curve=np.zeros([num_trials,num_agents])
    local_alignment=np.zeros([num_trials,del_steps])
    cohesion=np.zeros([num_trials,del_steps])
    
    
    #interaction
    
    for i_trial in range(num_trials):
        
        
        for i in range(num_agents):#reset g matrix to not mix the actions of current trial with past trials.
            agent_list[i].g_matrix=np.zeros((agent_list[i].num_actions, agent_list[i].num_percepts), dtype=np.float64) 
            
        reward_trial_list=np.zeros(num_agents)#reset reward list and test of rewards
        reward_list=np.zeros(num_agents)
        test_reward=np.zeros(num_agents)
        
      
        next_observation=env.reset()#reset positions of all agents and take percept for first agent.
        
        
        for t in range(del_steps):
            counter_align=0
            local_align=0
            cohes=0
            for i in range(num_agents):
                action=agent_list[i].deliberate_and_learn(next_observation,reward_list[i])
                next_observation,reward_list[i]=env.move(i,action,test_reward[i])
                if reward_list[i]==1:
                    test_reward[i]=1
                      
                reward_trial_list[i]=test_reward[i]
                
                cohes+=len(env.allagents_neighbours[i])#number of neighbours inside sensory range of agent i.
                counter_align+=env.speeds[i]  
                if len(env.allagents_neighbours[i])==0:
                    local_align+=0
                else:
                    local_align+=abs(np.sum(env.speeds[np.asarray(env.allagents_neighbours[i])])/len(env.allagents_neighbours[i]))
        
            alignment[i_trial,t]=counter_align/num_agents
            local_alignment[i_trial,t]=local_align/num_agents
            cohesion[i_trial,t]=cohes/num_agents
            
        learning_curve[i_trial]=reward_trial_list
#        if i_trial%1000==0:
#            print(i_trial)
#    
    alignment_pop[population]=np.sum(abs(alignment),axis=1)/del_steps
    local_alignment_pop[population]=np.sum(local_alignment,axis=1)/del_steps 
    cohesion_pop[population]=np.sum(cohesion,axis=1)/del_steps
    learning_pop[population]=np.sum(learning_curve,axis=1)/num_agents
#    file = open( 'save_agents_d21_pop={0}'.format(population)+'.txt', "wb" )
#    pickle.dump((agent_list,i_trial),file)
#    file.close()
#     
#
#
#np.savetxt('learning_curve_d21.txt', learning_pop)
#np.savetxt('cohesion_d21.txt', cohesion_pop)
#np.savetxt('alignment_d21.txt', alignment_pop)
#np.savetxt('local_alignment_d21.txt', local_alignment_pop)
#
