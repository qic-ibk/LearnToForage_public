# -*- coding: utf-8 -*-
"""
Copyright 2020 Andrea López Incera.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Andrea López Incera, used and analysed in, 

'Development of swarm behavior in artificial learning agents that adapt to different foraging environments.'
Andrea López-Incera, Katja Ried, Thomas Müller and Hans J. Briegel.

This piece of code includes all the methods and classes needed to perform the statistical analysis of foraging models.

"""
import numpy as np
import scipy
import scipy.stats as sts
import numpy.ma as ma
import scipy.optimize as opt
import collections

class foragingmodels(object):
    
    
    def __init__(self, raw_data):
        """Initialization. Argument is the array with the step lengths."""
        
        self.raw_data = raw_data;

    #probability distributions for the four models: Brownian motion (RW), composite random walk (CRW), levy walk and composite correlated random walks (CCRW).
    def exp_distr(self,lambd,data_point):
        """Exponential probability distribution."""
        return (1-np.exp(-lambd))*np.exp(-lambd*(data_point-1))

    def CRW_distr(self,parameters,data_point):
        """PDF for composite random walks. Returns pdf for each data point considering all data points come from an exponential distribution starting at xmin=1 (min value of the observed data)."""
        gamma_int,gamma_ext,p= parameters
        pdf_vals=p*(1-np.exp(-gamma_int))*np.exp(-gamma_int*(data_point-1))+(1-p)*(1-np.exp(-gamma_ext))*np.exp(-gamma_ext*(data_point-1))
        return pdf_vals
    
    def powerlaw(self,parameters,data_point):
        """PDF for powerlaw distribution, xmin=1 (min value of the observed data)."""
       
        alpha=parameters
        
        #power law pdf to fit the data above xmin. Renormalized.
        pdf_vals=(1./scipy.special.zeta(alpha,1))*data_point**(-alpha)
        
        return pdf_vals
    
    def CCRW(self,sample_size,parameters):
        """Given the parameters from the MLE, it generates samples of size sample_size from this distribution in order to get an approximation of the pdf."""
        
        delta,lamint,lamext,pint,pext=parameters
        samp=np.zeros(sample_size)
        
        current_mode=np.random.choice(2,p=[delta,1-delta]) #0 indicates the intensive mode and 1 the extensive.
        if current_mode:
            samp[0]=np.random.choice(np.arange(1,100000),p=(1-np.exp(-lamext))*np.exp(-lamext*(np.arange(1,100000)-1)))
        else:
            samp[0]=np.random.choice(np.arange(1,100000),p=(1-np.exp(-lamint))*np.exp(-lamint*(np.arange(1,100000)-1)))
        
        for i in range(1,sample_size):
            if current_mode:#previous mode was extensive
                current_mode=np.random.choice(2,p=[1-pext,pext])#with prob pext it stays in the extensive mode (1).
                if current_mode:
                    samp[i]=np.random.choice(np.arange(1,100000),p=(1-np.exp(-lamext))*np.exp(-lamext*(np.arange(1,100000)-1)))
                else:
                    samp[i]=np.random.choice(np.arange(1,100000),p=(1-np.exp(-lamint))*np.exp(-lamint*(np.arange(1,100000)-1)))
            else:#previous mode was intensive
                current_mode=np.random.choice(2,p=[pint,1-pint])
                if current_mode:
                    samp[i]=np.random.choice(np.arange(1,100000),p=(1-np.exp(-lamext))*np.exp(-lamext*(np.arange(1,100000)-1)))
                else:
                    samp[i]=np.random.choice(np.arange(1,100000),p=(1-np.exp(-lamint))*np.exp(-lamint*(np.arange(1,100000)-1)))
        
        return samp
    
    #maximum likelihood estimations for each model
    
    def MLE_exp(self,lam_init):
        """Computes the maximum likelihood estimators to get the best fit with the exponential distribution. The
        minimum value of the step length is the observed value, i.e. 1. This estimates the decay rate.
        Input:initial parameter for optimization.
        Output: decay rate, std of decay rate, log likelihood, akaike value."""
       
        def prob_distr_exp(parameters):#discrete exponential
            """Returns pdf for each data point considering all data points come from an exponential distribution starting at xmin=1 (min value of the observed data)."""
            lam = parameters
            pdf_vals=(1-np.exp(-lam))*np.exp(-lam*(self.raw_data-1))
            return pdf_vals
        
        def neg_log_likelihood_exp(parameters):
            """Given the parameters, it returns the value of the negative of the log likelihood function."""
            pdf_vals = prob_distr_exp(parameters)
            ln_pdf_vals = np.log(pdf_vals)
            return -np.sum(ln_pdf_vals[np.isfinite(ln_pdf_vals)])
        
        #optimization
        bnds = [(0.00000001, None)]
        params_init = np.array([lam_init])
        results_op = opt.minimize(neg_log_likelihood_exp, params_init,bounds=bnds)
        
        #get results of the MLE. 
        lam_MLE= results_op.x
        vcv=results_op.hess_inv.matmat(np.eye(len(params_init)))
        sdev_lam_MLE=np.sqrt(vcv[0,0])
        
        #model parameters: lambda, xmin.
        
        return lam_MLE,sdev_lam_MLE,-results_op.fun, 2*2+2*results_op.fun
    
    def MLE_CRW(self,par_int_init,gamma_ext_init,p_init):
        """Computes the maximum likelihood estimators to get the best fit with a mix of exponential distributions. The
        minimum value of the step length is the observed value, i.e. 1. This estimates the decay rate of both exponentials and the probability of taking each distribution.
        Input: initial values for the parameters.
        Output: estimated parameters, std of estimated parameters, log likelihood, akaike value."""
        
        def prob_distr_crw(parameters):#discrete exponential
            """Returns pdf for each data point considering all data points come from an exponential distribution starting at xmin=1 (min value of the observed data)."""
            par_int,gamma_ext,p= parameters
            gamma_int=gamma_ext+par_int
            pdf_vals=p*(1-np.exp(-gamma_int))*np.exp(-gamma_int*(self.raw_data-1))+(1-p)*(1-np.exp(-gamma_ext))*np.exp(-gamma_ext*(self.raw_data-1))
            return pdf_vals
        
        def neg_log_likelihood_crw(parameters):
            """Given the parameters, it returns the value of the negative of the log likelihood function."""
            pdf_vals = prob_distr_crw(parameters)
            ln_pdf_vals = np.log(pdf_vals)
            return -np.sum(ln_pdf_vals[np.isfinite(ln_pdf_vals)])
        
        #optimization
        bnds = [(0.00000001, None),(0.00000001, None),(0.00000001, 1)]
        params_init = np.array([par_int_init,gamma_ext_init,p_init])
        results_op = opt.minimize(neg_log_likelihood_crw, params_init,bounds=bnds)
        
        #get results of the MLE.
        par_int_MLE,gamma_ext_MLE,p_MLE=results_op.x
        gamma_int_MLE=gamma_ext_MLE+par_int_MLE
        vcv=results_op.hess_inv.matmat(np.eye(len(params_init)))
        sdev_gamma_ext_MLE=np.sqrt(vcv[1,1])
        sdev_gamma_int_MLE=np.sqrt(vcv[0,0])+np.sqrt(vcv[1,1])
        sdev_p_MLE=np.sqrt(vcv[2,2])
        
        
        #model parameters: lambda_ext, lambda_int, probability, xmin.
        
        return gamma_int_MLE,gamma_ext_MLE,p_MLE,sdev_gamma_int_MLE,sdev_gamma_ext_MLE,sdev_p_MLE,-results_op.fun, 2*4+2*results_op.fun
    
    def MLE_powerlaw(self,alpha_init):
        """Computes the maximum likelihood estimators to get the best fit with a powerlaw distribution. 
        Input: initial value for the exponent.
        Output: estimated parameters, std of estimated parameters, log likelihood, akaike value."""
        
        def prob_distr_pl(parameters):#discrete levy walk
            """Returns an array with the pdf values of each data point."""
            alpha=parameters
            
            #power law pdf to fit the data above xmin (trim_data). Renormalized.
            pdf_vals=(1./scipy.special.zeta(alpha,1))*self.raw_data**(-alpha)
            
            return pdf_vals
            
        def neg_log_likelihood_pl(parameters):
            """Given the parameters, it returns the value of the negative of the log likelihood function."""
            pdf_vals = prob_distr_pl(parameters)
            ln_pdf_vals = np.log(pdf_vals)
        
            return -np.sum(ln_pdf_vals[np.isfinite(ln_pdf_vals)])


            
        #optimization
        bnds = [(1.00000001, None)]
        params_init = np.array([alpha_init])
        results_op = opt.minimize(neg_log_likelihood_pl, params_init,bounds=bnds)
        vcv=results_op.hess_inv.matmat(np.eye(len(params_init)))
            
        #get results of the MLE.  
        xmin_MLE=1
        alpha_MLE=results_op.x
        sdev_alpha_MLE=np.sqrt(vcv[0,0])
        #model parameters :2
        
        
        return xmin_MLE,alpha_MLE,sdev_alpha_MLE,-results_op.fun,2*2+2*results_op.fun
    
    def MLE_CCRW(self,delta_init,a_init,lamext_init,pint_init,pext_init):
        """Computes the maximum likelihood estimators to get the best fit with a CCRW as a Hidden Markov chain model. The
        minimum value of the step length is the observed value, i.e. 1. This estimates the decay rate of both exponentials, the probability to start in the intensive mode and the transition probabilities
        from one mode to the other.
        Input: initial values for the parameters.
        Output: estimated parameters, std of estimated parameters, log likelihood, akaike value."""
        
            
        def neg_log_likelihood_ccrw(parameters):#discrete exponential
            """Given the parameters, it returns the value of the negative of the log likelihood function."""
            delta,a,lamext,pint,pext= parameters
            lamint=a+lamext#reparametrization so that lambda_int>lambda_ext. 
            
            matrix1=np.matrix([[delta*self.exp_distr(lamint,self.raw_data[0]),(1-delta)*self.exp_distr(lamext,self.raw_data[0])]])#delta P1
            
            w1=np.sum(matrix1)
            likelihood=np.log(w1)
            phi=(1.0/w1)*matrix1
            for i in self.raw_data[1:]:
                v=phi*np.matrix([[pint*self.exp_distr(lamint,i),(1-pint)*self.exp_distr(lamext,i)],[(1-pext)*self.exp_distr(lamint,i),pext*self.exp_distr(lamext,i)]])
                u=np.sum(v)
                likelihood=likelihood+np.log(u)
                phi=(1.0/u)*v
            
            return -1.0*likelihood
           
        
        #optimization
        bnds = [(0.00000001, 1),(0.00000001, 1),(0.00000001, 0.5),(0.00000001, 0.99999),(0.00000001, 0.99999)]
        params_init = np.array([delta_init,a_init,lamext_init,pint_init,pext_init])
        results_op = opt.minimize(neg_log_likelihood_ccrw, params_init,bounds=bnds)
        
        #get results of the MLE. Compute AIC value.
        delta_MLE,a_MLE,lamext_MLE,pint_MLE,pext_MLE = results_op.x
        lamint_MLE=lamext_MLE+a_MLE
        vcv=results_op.hess_inv.matmat(np.eye(len(params_init)))
        sdev_delta_MLE=np.sqrt(vcv[0,0])
        sdev_lamint_MLE=np.sqrt(vcv[1,1])+np.sqrt(vcv[2,2])
        sdev_lamext_MLE=np.sqrt(vcv[2,2])
        sdev_pint_MLE=np.sqrt(vcv[3,3])
        sdev_pext_MLE=np.sqrt(vcv[4,4])
        
        #model parameters: parameters_MLE and xmin=1.
        
        return delta_MLE,lamint_MLE,lamext_MLE,pint_MLE,pext_MLE ,sdev_delta_MLE,sdev_lamint_MLE,sdev_lamext_MLE,sdev_pint_MLE,sdev_pext_MLE,-results_op.fun,2*(1+len(params_init))+2*results_op.fun
    
    def lnlikel_raw(self):
        """It computes the loglikelihood of the observed data, i.e. the probability of each data point is just its observed frequency."""
        freq=collections.Counter(np.sort(self.raw_data))
        p_exp=np.zeros(len(self.raw_data))
        c=0
        for i in np.sort(self.raw_data):
            p_exp[c]=freq[i]/len(self.raw_data)
            c+=1
        return np.sum(np.log(p_exp))
    
    #goodness of fit tests
    
    def logratio(self,distribution,parameters_mle):
        """It computes a goodness of fit test for the chosen distribution. Not valid for CCRW.
        Input: distribution type, estimated parameters.
        Output: log_ratio,variance,pvalue."""
        
        #computation of experimental probabilities (data in order).
        freq=collections.Counter(self.raw_data)
        p_exp=np.zeros(len(self.raw_data))
        c=0
        for i in np.sort(self.raw_data):
            p_exp[c]=freq[i]/len(self.raw_data)
            c+=1
        
        #computation of theoretical probabilities.
        if distribution=='exponential': 
            lambd=parameters_mle
            p_th=(1-np.exp(-lambd))*np.exp(-lambd*(np.sort(self.raw_data)-1))
            log_ratio=np.sum(np.log(p_exp)-np.log(p_th))
            
            difference=(np.log(p_exp)-np.log(p_th))-(np.mean(np.log(p_exp))-np.mean(np.log(p_th)))
            variance=(1/len(self.raw_data))*np.sum(difference*difference)
            pvalue=np.abs(scipy.special.erfc(log_ratio/np.sqrt(2*len(self.raw_data)*variance)))
            
            return pvalue
        
        if distribution=='powerlaw':
            xmin,alpha=parameters_mle
            trim_data=self.raw_data[self.raw_data>=xmin]
            
            p_th=(1./scipy.special.zeta(alpha,xmin))*np.sort(trim_data)**(-alpha)
            
            log_ratio=np.sum(np.log(p_exp)-np.log(p_th))
            
            difference=(np.log(p_exp)-np.log(p_th))-(np.mean(np.log(p_exp))-np.mean(np.log(p_th)))
            variance=(1/len(self.raw_data))*np.sum(difference*difference)
            pvalue=np.abs(scipy.special.erfc(log_ratio/np.sqrt(2*len(self.raw_data)*variance)))
            
            return pvalue
        
        if distribution=='CRW': 
            gamma_int,gamma_ext,p=parameters_mle
            p_th=p*(1-np.exp(-gamma_int))*np.exp(-gamma_int*(self.raw_data-1))+(1-p)*(1-np.exp(-gamma_ext))*np.exp(-gamma_ext*(self.raw_data-1))
            log_ratio=np.sum(np.log(p_exp)-np.log(p_th))
            
            difference=(np.log(p_exp)-np.log(p_th))-(np.mean(np.log(p_exp))-np.mean(np.log(p_th)))
            variance=(1/len(self.raw_data))*np.sum(difference*difference)
            pvalue=np.abs(scipy.special.erfc(log_ratio/np.sqrt(2*len(self.raw_data)*variance)))
            
            return pvalue
        
    #functions needed for the computation of pseudo residuals.
    def cdf(self,lamint,lamext,data):
        """Cumulative distribution function for the discrete exponential. It returns an array of size nx2, where the
        first column contains the cdf of the intensive (with lambda intensive) exponential, and the second column contains
        the cdf of the extensive exponential."""
        
        dist_int=self.exp_distr(lamint,np.arange(1,max(data)+1))
        dist_ext=self.exp_distr(lamext,np.arange(1,max(data)+1))
        
        cdf=np.zeros([len(data),2])
        c=0
        for i in data:
            cdf[c,0]=np.sum(dist_int[:int(i)])
            cdf[c,1]=np.sum(dist_ext[:int(i)])
            c+=1
        return cdf
    
    def lalphabeta(self,parameters_mle):
        """Computes the log (log of each element) of the matrices of forward probabilities alpha_t and backward probabilities beta_t, for all the t 
        (where t is the tth data point). The parameters are the ones obtained with the maximum likelihood estimation."""
        
        delta,lamint,lamext,pint,pext=parameters_mle
        lalpha=np.zeros([len(self.raw_data),2])
        lbeta=np.zeros([len(self.raw_data),2])
        #logarithms are computed and matrices are rescaled to avoid overflow.
        foo=np.matrix([[delta*self.exp_distr(lamint,self.raw_data[0]),(1-delta)*self.exp_distr(lamext,self.raw_data[0])]])
        lscale=np.log(np.sum(foo))
        foo=(1.0/np.sum(foo))*foo
        lalpha[0,:]=np.log(foo)+lscale
        
        for i in range(1,len(self.raw_data)):
            foo=foo*np.matrix([[pint*self.exp_distr(lamint,self.raw_data[i]),(1-pint)*self.exp_distr(lamext,self.raw_data[i])],[(1-pext)*self.exp_distr(lamint,self.raw_data[i]),pext*self.exp_distr(lamext,self.raw_data[i])]])
            lscale=lscale+np.log(np.sum(foo))
            foo=(1.0/np.sum(foo))*foo
            lalpha[i,:]=np.log(foo)+lscale
        
        foo=np.matrix([[0.5],[0.5]])
        lscale=np.log(2.0)
        reverted_data=self.raw_data[::-1]
        c=len(self.raw_data)-2
        for i in reverted_data[:(len(self.raw_data)-1)]:
            foo=np.matrix([[pint*self.exp_distr(lamint,i),(1-pint)*self.exp_distr(lamext,i)],[(1-pext)*self.exp_distr(lamint,i),pext*self.exp_distr(lamext,i)]])*foo
            lbeta[c,:]=np.matrix.transpose(np.log(foo)+lscale)
            foo=(1.0/np.sum(foo))*foo
            lscale=lscale+np.log(np.sum(foo))
            c=c-1
        
        return lalpha,lbeta
    
    #pseudoresiduals and GOF test.
    def pseudores(self,parameters_mle):
        """Given the parameters of the MLE, this function computes the uniform pseudo residuals and a KS test for uniformity on the 
        mid-pseudoresiduals. Since our case deals with discrete prob. distr., this method computes the lower and upper residuals that limit the 
        pseudo residual interval. Also the mid-residual is given."""
        
        lalpha,lbeta=self.lalphabeta(parameters_mle)
        delta,lamint,lamext,pint,pext=parameters_mle
        
        #include alpha_0, to compute the forward prob. when t=1.
        lalpha=np.concatenate((np.log(np.array([delta,(1-delta)]))[None,:],lalpha),axis=0)
        #reescale it to avoid overflow and recover alpha and beta (rescaled) matrices.
        alpha_for=np.exp(lalpha-np.amax(lalpha,axis=1)[:,None])
        beta_back=np.exp(lbeta-np.amax(lbeta,axis=1)[:,None])
        #multiplication alpha*gamma (array (n+1)x2).
        alpha_gamma=np.concatenate(((alpha_for[1:,0]*pint+alpha_for[1:,1]*(1-pext))[:,None],(alpha_for[1:,0]*(1-pint)+alpha_for[1:,1]*pext)[:,None]),axis=1)
        alpha_gamma=np.concatenate((alpha_for[0,:][None,:],alpha_gamma),axis=0)
        #weights wi (array nx2).
        w=(1/np.sum(alpha_gamma[:(len(alpha_gamma)-1)]*beta_back,axis=1))[:,None]*(alpha_gamma[:(len(alpha_gamma)-1)]*beta_back)
        #array of size len(data_tofit) with all the pseudo residuals (one interval per data point).
        u_plus=np.sum((self.cdf(lamint,lamext,self.raw_data)*w),axis=1) #array multiplication, i.e. element by element. Then the two columns are summed up to get the pseudo residual.
        u_minus=np.sum((self.cdf(lamint,lamext,self.raw_data-1)*w),axis=1)
        
        #mid pseudo residuals
        u_mid=0.5*(u_plus+u_minus)
        
        u_prob=u_plus-u_minus
        
        #KS test to check if u_mid are uniformly distributed.
        D,pvalue=sts.kstest(u_mid,'uniform')
            
        return u_plus,u_minus,u_mid,u_prob,D,pvalue
            
          