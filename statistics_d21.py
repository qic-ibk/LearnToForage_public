# -*- coding: utf-8 -*-
"""
Copyright 2020 Andrea López Incera.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Andrea López Incera, used and analysed in, 

'Development of swarm behavior in artificial learning agents that adapt to different foraging environments.'
Andrea López-Incera, Katja Ried, Thomas Müller and Hans J. Briegel.

This piece of code takes the ---previously generated--- trajectories for the analysis of foraging models, and 
perform the whole analysis, including MLE of the parameters, Akaike weights and GOF tests.
"""

import numpy as np
import scipy
import scipy.stats as sts
import numpy.ma as ma
import scipy.optimize as opt
import pickle
import collections

import stat_study_models


def steps(trajectory,world_size):
    """Given a trajectory, it computes the step lengths from the world positions of the agent.
    input:  array with the positions of the agent through the trajectory, size of the circular world.
    output: array with the (ordered) step lengths."""
    steps=[]
    
    unfolded_positions=np.copy(trajectory)
    
    for pos_index in range(1,len(unfolded_positions)):
        if (unfolded_positions[pos_index]-unfolded_positions[pos_index-1])<-4:
            unfolded_positions[pos_index]+=round(abs(unfolded_positions[pos_index]-unfolded_positions[pos_index-1])/world_size)*world_size
        if (unfolded_positions[pos_index]-unfolded_positions[pos_index-1])>4:
            unfolded_positions[pos_index]-=round(abs(unfolded_positions[pos_index]-unfolded_positions[pos_index-1])/world_size)*world_size
    
    step_counter=unfolded_positions[1]-unfolded_positions[0]
    for pos_index in range(2,len(unfolded_positions)):
        if (unfolded_positions[pos_index]-unfolded_positions[pos_index-1])==(unfolded_positions[pos_index-1]-unfolded_positions[pos_index-2]):
            step_counter+=(unfolded_positions[pos_index]-unfolded_positions[pos_index-1])
        else:
            steps.append(abs(step_counter))
            step_counter=(unfolded_positions[pos_index]-unfolded_positions[pos_index-1])
    
    return np.array(steps)


#parameters and imported data.
record_pos_end=pickle.load(open( "positions_traj_levy_d21.txt", "rb" ))
#record_pos_end=pickle.load(open( "positions_traj_levy_d4.txt", "rb" )) #import different data set for analysis of other trajectories, such as the ones obtained with agents trained with dF=4.
num_agents=60
world_size=500
num_pop=10

#matrices to store results.
exp_results=np.zeros([num_agents*num_pop,5])
CRW_results=np.zeros([num_agents*num_pop,9])
PL_results=np.zeros([num_agents*num_pop,6])
CCRW_results=np.zeros([num_agents*num_pop,13])

res_plus=2*np.ones([num_agents*num_pop,30000])#since the number of steps is different for every agent, we set the matrices
#to have a size of 30000. There will be some entries with value 2 that we need to take away from the matrix to analyze the data. We chose value 2 since it cannot be a pseudoresidual.
res_minus=2*np.ones([num_agents*num_pop,30000])
res_prob=2*np.ones([num_agents*num_pop,30000])

AIC_weights=np.zeros([num_agents*num_pop,4])

log_likel_data=np.zeros(num_agents*num_pop)


#compute the statistics for each agent's trajectory.
c=0
for pop in range(num_pop):
    for ag in range(num_agents):

        data_tofit=steps(record_pos_end[pop][:,ag],world_size)
        formod=stat_study_models.foragingmodels(data_tofit)
        AIC=np.zeros(4)
        
        exp_results[c,:4]=formod.MLE_exp(0.1)
        exp_results[c,4]=formod.logratio('exponential',exp_results[c,0])
        AIC[0]=exp_results[c,3]
        
        CRW_results[c,:8]=formod.MLE_CRW(0.1,0.0003,0.3)
        CRW_results[c,8]=formod.logratio('CRW',[CRW_results[c,0],CRW_results[c,1],CRW_results[c,2]])
        AIC[1]=CRW_results[c,7]
        
        PL_results[c,:5]=formod.MLE_powerlaw(2.0)
        PL_results[c,5]=formod.logratio('powerlaw',[PL_results[c,0],PL_results[c,1]])
        AIC[2]=PL_results[c,4]
        
        CCRW_results[c,:12]=formod.MLE_CCRW(0.1,0.2,0.01,0.7,0.002)
        res_plus[c,:len(data_tofit)],res_minus[c,:len(data_tofit)],u_mid,res_prob[c,:len(data_tofit)],CCRW_results[c,12],pvalue=formod.pseudores([CCRW_results[c,0],CCRW_results[c,1],CCRW_results[c,2],CCRW_results[c,3],CCRW_results[c,4]])
        AIC[3]=CCRW_results[c,11]
        
        AIC_weights[c]=np.exp(-0.5*(AIC-min(AIC)))/np.sum(np.exp(-0.5*(AIC-min(AIC))))
        
        log_likel_data[c]=formod.lnlikel_raw()
        c+=1
        

        

#np.savetxt('exp_results_d21.txt',exp_results)
#np.savetxt('CRW_results_d21.txt',CRW_results)
#np.savetxt('PL_results_d21.txt',PL_results)
#np.savetxt('CCRW_results_d21.txt',CCRW_results)
#
#np.savetxt('res_plus_d21.txt',res_plus)
#np.savetxt('res_minus_d21.txt',res_minus)
#np.savetxt('res_prob_d21.txt',res_prob)
#
#np.savetxt('AIC_weights_d21.txt',AIC_weights)
#
#np.savetxt('log_likel_data_d21.txt',log_likel_data)
        