#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import cvxpy as cp
import scipy
import mosek
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


# In[51]:


n = 10 #number of stocks
T_train = 100 # training period
T_test = 100 #testing period
T_truetest = 1000 # out of sample testing period


# In[52]:


#parameters for generating normal returns
mu = 0.0 + np.linspace(-0.1, 0.5, n)
cov = np.ones((n,n))+np.eye(n)

cov = 1 * cov


# In[53]:


#generating in-sample returns from Gaussian
def generate_data():
    mean = mu
    covariance = cov
    data = np.random.multivariate_normal(mean, covariance, T_train + T_test + T_truetest)
    #data = np.vstack((data, np.random.multivariate_normal(mean , covariance, T_test)))
    return data


# In[54]:


#generating mask --- MCAR
def missing(miss_prob=0.5):
    #''True'' represents missing
    mask = np.random.choice([True, False], size = (T_train,n), p = [miss_prob, 1-miss_prob])
    mask = np.vstack((mask, np.full((T_test + T_truetest,n),False))) #no missing value for testing period
    return mask


# In[55]:


#prior for the paramter \mu, use flat prior
mu_p = np.repeat(0.0,n)
covp_inv = np.diag(np.full(n,0.0))


# In[56]:


def individualposterior(data, mask, mu0, cov0_inv):
    
    meanlist = []
    covariancelist = []
    num_posteriors = T_test + 1
    
    covariance_mu_inv = np.copy(cov0_inv)
    mean_mu = np.matmul(covariance_mu_inv,mu0)
     
    for t in range(T_train + T_test):
        
        if not np.all(mask[t,:] == np.full(n,True)):
            
            ind = (mask[t,:] == np.full(n,False))
            cov_t = cov[np.ix_(ind,ind)]
            cov_inv_expand = np.zeros((n,n))
            cov_inv_expand[np.ix_(ind,ind)] = np.linalg.inv(cov_t)

            covariance_mu_inv += cov_inv_expand

            mean_t = np.zeros(n)
            mean_t[ind] = data[t,ind]
            mean_mu += np.matmul(cov_inv_expand,mean_t)
        
        if t >= (T_train - 1):
            covariance_mu = np.linalg.inv(covariance_mu_inv)
            covariancelist.append(covariance_mu)
            meanlist.append(np.matmul(covariance_mu, mean_mu))
    return meanlist,covariancelist  


# In[57]:


def consensusforwardkl(meanlist,covariancelist,delta_r,prediction,rho):
    
    num_posteriors = len(meanlist)
    n = len(meanlist[0])
    # Define optimization variables
    weights = cp.Variable(num_posteriors)
    gamma = cp.Variable(n)
    # v, s, vt = np.linalg.svd(covariancelist[0].values)
    
    s, v = np.linalg.eigh(covariancelist[0])
    
    s_list = []
    
    # s_list.append(s)
    
   
    
    # s, v = np.linalg.eigh(covariancelist[1].values)
    # s_list.append(s)
    
    # for i in np.arange(1,num_posteriors,1):
    for i in range(num_posteriors):
        s_temp = np.zeros(n)
        for j in range(n):
            s_temp[j] = np.inner(v[:,j],np.matmul(covariancelist[i],v[:,j]))
        s_list.append(s_temp)
        
    c = np.zeros((num_posteriors,n))
    
    for i in range(num_posteriors):
        for j in range(n):
            c[i, j] = np.inner(v[:,j],meanlist[i])/s_list[i][j]
            
        
    sv_matrix = np.array(s_list)     
    inverse_sv = 1.0 / sv_matrix
    
    #objective_fun = [cp.power(cp.sum(cp.multiply(inverse_sv[:,j], weights)),-1) for j in range(n)]
    
    obj = cp.sum([gamma[j] for j in range(n)])#sum(gamma_j)
    
    # Run optimization
    objective = cp.Minimize(obj)
    delta = delta_r * max([np.abs(c[-1,j]/inverse_sv[-1,j] - v[:,j].dot(prediction)) for j in range(n)])
    constraints = [weights >= 0,
                   cp.sum(weights) == 1]
    for j in range(n):
        constraints.append(cp.sum(cp.multiply(c[:,j],weights)) <= (delta/rho + v[:,j].dot(prediction)) * cp.sum(cp.multiply(inverse_sv[:,j], weights)))
        constraints.append(cp.sum(cp.multiply(c[:,j],weights)) >= (-delta/rho + v[:, j].dot(prediction)) * cp.sum(cp.multiply(inverse_sv[:,j], weights)))
        #constraints.append(4 +  cp.power(cp.sum(cp.multiply(inverse_sv[:,j], weights))-gamma[j],2)<= cp.power(cp.sum(cp.multiply(inverse_sv[:,j], weights))+gamma[j],2))   
        A = np.zeros((2,num_posteriors))
        B = np.zeros((2,n))
        B[1,j] = 1 
        for i in range(num_posteriors):
            A[1,i] = inverse_sv[i,j]
        C = np.zeros(2)
        C[0] = 2
        constraints.append(cp.SOC(A[1,:]@weights + B[1,:]@gamma, A @ weights - B @ gamma + C))
                           
                           
    prob = cp.Problem(objective, constraints)
   
    prob.solve(solver=cp.MOSEK)
    
    solution = weights.value
    #print(solution)
    #print(solution)
    
    final_sigma = scipy.linalg.inv(sum([solution[i] * scipy.linalg.inv(covariancelist[i]) for i in range(num_posteriors)]))
    final_mu = final_sigma.dot(sum([solution[i] * np.inner(scipy.linalg.inv(covariancelist[i]), meanlist[i]) for i in range(num_posteriors)]))
    
    return solution, final_mu, final_sigma


# In[58]:


def consensuswasserstein(meanlist,covariancelist,delta_r,prediction,rho,set_choice):
    
    num_posteriors = len(meanlist)
    n = len(meanlist[0])
    
    weights = cp.Variable(2)
    s, v = np.linalg.eigh(covariancelist[0])
    
        
    Sigma1 = covariancelist[0]
    Sigma2 = covariancelist[-1]
    temp = sqrtm(Sigma2) @ Sigma1 @ sqrtm(Sigma2)
    Psi = sqrtm(Sigma2) @ np.real(scipy.linalg.inv(sqrtm(temp))) @ sqrtm(Sigma2)
    
    P = np.zeros((2,2))
    P[0,0] = np.trace(Sigma1)
    P[1,1] = np.trace(Sigma2)
    P[0,1] = np.trace(Sigma1 @ Psi)
    P[1,0] = np.trace(Sigma1 @ Psi)
   
    obj = cp.quad_form(weights, P)
    
    delta = delta_r * np.linalg.norm(meanlist[-1] - prediction)
    constraints = [weights >= 0,
                   cp.sum(weights) == 1]
    
    if set_choice == 'l2-norm':
        constraints.append(cp.norm(weights[0]*meanlist[0]+weights[1]*meanlist[1]-prediction)<=delta/rho)
    if set_choice == 'Vz-l1-norm':
        constraints.append(cp.norm(v @ (weights[0]*meanlist[0]+weights[1]*meanlist[1]-prediction),"inf") <= delta/rho)
        
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.MOSEK)
    
    solution = weights.value
   
    final_mu = meanlist[0] * solution[0] + meanlist[-1] * solution[1]
    final_sigma = (solution[0]*np.eye(n) + solution[1] * Psi) @ Sigma1 @ (solution[0]*np.eye(n) + solution[1] * Psi)
    
    return solution, final_mu, final_sigma


# In[59]:


def consensuswasserstein_general(meanlist,covariancelist,delta_r, prediction, rho, set_choice):
    
    num_posteriors = len(meanlist)
    n = len(meanlist[0])
    
    # Define optimization variables
    weights = cp.Variable(num_posteriors)
    
    # v, s, vt = np.linalg.svd(covariancelist[0].values)
    
    s, v = np.linalg.eigh(covariancelist[0])
    
    s_list = []
    
    # for i in np.arange(1,num_posteriors,1):
    for i in range(num_posteriors):
        s_temp = np.zeros(n)
        for j in range(n):
            s_temp[j] = np.inner(v[:,j],np.matmul(covariancelist[i],v[:,j]))
        s_list.append(s_temp)
    
    sv_matrix = np.array(s_list)
    
    #P = np.zeros((num_posteriors,num_posteriors))
    #for i in range(num_posteriors): 
        #for j in range(num_posteriors):
            #P[i,j] = np.sum(np.multiply(np.sqrt(sv_matrix[i,:]),np.sqrt(sv_matrix[j,:])))
            
            
    #obj = 0
    #for i in range(num_posteriors): 
    #    for j in range(num_posteriors):
    #        obj += weights[i] * weights[j] * np.sum(np.multiply(np.sqrt(sv_matrix[i,:]),np.sqrt(sv_matrix[j,:])))
    
    
    #obj = cp.sum_squares(weights @ sv_matrix)
    obj = cp.sum_squares(sv_matrix.T @ weights.T)
    #obj = cp.quad_form(weights, P)
    delta = delta_r * np.linalg.norm(meanlist[-1] - prediction)
    constraints = [weights >= 0,
                   cp.sum(weights) == 1]
    temp = 0
    for i in range(num_posteriors):
        temp += weights[i]*meanlist[i]#\sum(\lambda_k\mu_k)
    if set_choice == 'l2-norm':
        constraints.append(cp.norm(temp-prediction)<=delta/rho)
    if set_choice == 'Vz-l1-norm':
        constraints.append(cp.norm(v @ (temp-prediction),"inf") <= delta/rho)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.MOSEK)

    
    solution = weights.value
    
    
    final_mu = np.zeros(n)
    for i in range(num_posteriors):
        final_mu += solution[i]*meanlist[i]
    final_sigma = np.zeros((n,n))
    for i in range(num_posteriors):
        for j in range(n):
            final_sigma += solution[i] * np.sqrt(sv_matrix[i,j]) * np.outer(v[:,j],v[:,j])
    final_sigma = final_sigma @ final_sigma
    
    return solution, final_mu, final_sigma


# In[60]:


def consensusbackwardkl(meanlist,covariancelist,delta_r,prediction,weight_init,epsilon,max_iter,rho,set_choice):#weight_init = np.full(num_posteriors,1/num_posteriors)
    
    num_posteriors = len(meanlist)
    n = len(meanlist[0])
    
    # Define optimization variables
    weights = cp.Variable(num_posteriors)
    
    s, v = np.linalg.eigh(covariancelist[0])
    
    mu_list = []
    
    obj1 = 0
    for i in range(num_posteriors):
        mean_k = meanlist[i]
        mu_list.append(mean_k)
        obj1 += weights[i]*cp.trace(np.outer(mean_k,mean_k)+covariancelist[i])
        
    mu_matrix = np.array(mu_list)
    
    weight_t = weight_init
    weight_former = np.zeros(num_posteriors)
    iteration = 0
    
    while np.linalg.norm(weight_t-weight_former) > epsilon and iteration < max_iter:
        
        weight_former = weight_t
        
        #obj2 = 2*np.inner(np.matmul(mu_matrix,weight_t),np.matmul(mu_matrix,weights))
        obj2 = 2*(np.inner(weight_t,np.inner(mu_matrix,mu_matrix))@ weights)
        
        for i in range(num_posteriors):
            for j in range(num_posteriors):
                obj3 = -weight_t[i]*weight_t[j]*np.inner(meanlist[i],meanlist[j])
        
        obj = obj1+obj2+obj3
    
        delta = delta_r * np.linalg.norm(meanlist[-1] - prediction)
        constraints = [weights >= 0,
                   cp.sum(weights) == 1]
        temp = 0
        for i in range(num_posteriors):
            temp += weights[i]*meanlist[i]#\sum(\lambda_k\mu_k)
        if set_choice == 'l1-norm': 
            constraints.append(cp.norm(temp-prediction,"inf") <= delta/rho)
        if set_choice == 'Vz-l1-norm':
            constraints.append(cp.norm(v @ (temp-prediction),"inf") <= delta/rho)
        #temp = 0
        #for j in range(n): #j:0,...n-1
        #    for i in range(num_posteriors):
        #        temp += weights[i]*meanlist[i][j]
        #    constraints.append(temp-prediction[j] <= delta)
        #    constraints.append(temp-prediction[j] >= -delta)
            
            
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.MOSEK)
    
        weight_t = weights.value
        iteration += 1
        #print("weight diff:",np.linalg.norm(weight_t-weight_former))
        #print("iteration:",iteration)
    
    final_mu = np.zeros(n)
    for i in range(num_posteriors):
        final_mu += weight_t[i]*meanlist[i]
        
    final_sigma = np.zeros((n,n))
    final_sigma_1 = np.zeros((n,n))
    for i in range(num_posteriors):
        final_sigma_1 += weight_t[i]*(np.outer(meanlist[i],meanlist[i])+covariancelist[i])
    final_sigma = final_sigma_1 - np.outer(final_mu,final_mu)
    
    return weight_t, final_mu, final_sigma


# In[61]:


def imputation(data, mask, final_mu,final_sigma):    
    m = 10 #number of multiply-imputed dataset
    total_time,num_stocks = data.shape
    completed_data = np.zeros((m,total_time,num_stocks))
    for k in range(m):
        data_copy = np.copy(data)
        unconditionalmean = np.random.multivariate_normal(final_mu, final_sigma)
        for t in range(T_train):
            if np.all(mask[t,:] == np.full(n,True)):
                data_copy[t,:] = unconditionalmean
            elif np.all(mask[t,:] == np.full(n,False)):
                pass
            else:
                ind_miss = (mask[t,:] == np.full(n,True))
                ind_obs = (mask[t,:] == np.full(n,False))
                len_miss = len(data_copy[t,ind_miss])
                data_copy[t,ind_miss] = np.zeros(len_miss)
                cov11 = cov[np.ix_(ind_miss,ind_obs)]
                cov12_inv = np.linalg.inv(cov[np.ix_(ind_obs,ind_obs)])
                missing_condi_mean = unconditionalmean[ind_miss] + np.matmul(np.matmul(cov11,cov12_inv),data_copy[t,ind_obs] - unconditionalmean[ind_obs])
                missing_condi_cov = cov[np.ix_(ind_miss,ind_miss)] - np.matmul(cov11,np.matmul(cov12_inv,np.matrix.transpose(cov11)))
                data_copy[t,ind_miss] = missing_condi_mean
        completed_data[k,:,:] = data_copy
    return completed_data


# In[62]:


def Greedy(data1):
    data = np.copy(data1)
    #data = data / 1000
    mean = np.mean(data[:T_train,:],axis = 0)
    weights = mean / np.linalg.norm(mean,2)
    #c_returns = 1.0
    returns = np.zeros(T_test)
    for t in np.arange(T_train,T_train+T_test,1):
        returns[t-T_train] = np.inner(data[t,:], weights)
        #c_returns *= 1.0 + returns[t-T_train]
    sharper = np.mean(returns)/np.std(returns)
    
    #o_returns = 1.0
    returns_o = np.zeros(T_truetest)
    for t in np.arange(T_train+T_test,T_train+T_test+T_truetest,1):
        returns_o[t-T_train - T_test] = np.inner(data[t,:], weights)
       # o_returns *= 1.0 + returns_o[t-T_train - T_test]
    o_sharper = np.mean(returns_o)/np.std(returns_o)
    
    return sharper, o_sharper,np.mean(returns),np.mean(returns_o)
    #return np.mean(returns),np.mean(returns_o)


# In[63]:


n_experiment = 100
m = 10
 
num_delta = 10
num_rho = 10

mreturn_i_complex = np.zeros((n_experiment,m,num_delta))
mreturn_o_complex = np.zeros((n_experiment,m,num_delta))

mreturn_i_complex_wb = np.zeros((n_experiment,m,num_delta))
mreturn_o_complex_wb = np.zeros((n_experiment,m,num_delta))

mreturn_i_complex_wb_general = np.zeros((n_experiment,m,num_delta))
mreturn_o_complex_wb_general = np.zeros((n_experiment,m,num_delta))

mreturn_i_complex_kl_back = np.zeros((n_experiment,m,num_delta))
mreturn_o_complex_kl_back = np.zeros((n_experiment,m,num_delta))


# In[64]:


def experiment(param, hyperparam):
    #param is the testing parameter at experiment: delta or rho
    #hyperparam is the hyperparamter at experiment: e.g. when testing delta, the hyperparameter rho that we use.
    
    if param == 'delta':

        for k in range(n_experiment):
            print(k)
            data = generate_data()
            mask = missing()
            meanlist,covariancelist = individualposterior(data, mask, mu_p, covp_inv)
    
            deltalist_complex  = np.linspace(0.000, 1.0, num = num_delta)
            rholist_complex = np.linspace(0.001,5.0, num = num_rho)#rho should not equal to 0.
        
            for i in range(num_delta):
        
                _,final_mu,final_sigma = consensusforwardkl(meanlist,covariancelist,deltalist_complex[i],meanlist[0],rholist_complex[hyperparam])
                completed_data = imputation(data,mask,final_mu,final_sigma)
        
                _,final_mu_wb,final_sigma_wb = consensuswasserstein(meanlist,covariancelist,deltalist_complex[i],meanlist[0],rholist_complex[hyperparam],'Vz-l1-norm')
                completed_data_wb = imputation(data,mask,final_mu_wb,final_sigma_wb)
        
                _,final_mu_wb_general,final_sigma_wb_general = consensuswasserstein_general(meanlist,covariancelist,deltalist_complex[i],meanlist[0],rholist_complex[hyperparam],'Vz-l1-norm')
                completed_data_wb_general = imputation(data,mask,final_mu_wb_general,final_sigma_wb_general)
        
                _,final_mu_kl_back,final_sigma_kl_back = consensusbackwardkl(meanlist,covariancelist,deltalist_complex[i],meanlist[0],np.full(len(meanlist),1/len(meanlist)),10**(-14),20,rholist_complex[hyperparam],'Vz-l1-norm')
                completed_data_kl_back = imputation(data,mask,final_mu_kl_back,final_sigma_kl_back)
        
                for j in range(m):
                    _, _,mreturn_i_complex[k,j,i], mreturn_o_complex[k,j,i]  = Greedy(completed_data[j])
                    _, _,mreturn_i_complex_wb[k,j,i], mreturn_o_complex_wb[k,j,i]  = Greedy(completed_data_wb[j])
                    _, _,mreturn_i_complex_wb_general[k,j,i], mreturn_o_complex_wb_general[k,j,i]  = Greedy(completed_data_wb_general[j])
                    _, _,mreturn_i_complex_kl_back[k,j,i], mreturn_o_complex_kl_back[k,j,i]  = Greedy(completed_data_kl_back[j])
                
        sds_r = np.zeros((n_experiment * m ,  num_delta))
        for i in range(num_delta):
            sds_r[:,i] = mreturn_i_complex[:,:,i].flatten()
    
        sds_o_r = np.zeros((n_experiment * m ,  num_delta))
        for i in range(num_delta):
            sds_o_r[:,i] = mreturn_o_complex[:,:,i].flatten()

        sds_r_wb = np.zeros((n_experiment * m ,  num_delta))
        for i in range(num_delta):
            sds_r_wb[:,i] = mreturn_i_complex_wb[:,:,i].flatten()

        sds_o_r_wb = np.zeros((n_experiment * m ,  num_delta))
        for i in range(num_delta):
            sds_o_r_wb[:,i] = mreturn_o_complex_wb[:,:,i].flatten()

        sds_r_wb_general = np.zeros((n_experiment * m ,  num_delta))
        for i in range(num_delta):
            sds_r_wb_general[:,i] = mreturn_i_complex_wb_general[:,:,i].flatten()

        sds_o_r_wb_general = np.zeros((n_experiment * m ,  num_delta))
        for i in range(num_delta):
            sds_o_r_wb_general[:,i] = mreturn_o_complex_wb_general[:,:,i].flatten()
    
        sds_r_kl_back = np.zeros((n_experiment * m ,  num_delta))
        for i in range(num_delta):
            sds_r_kl_back[:,i] = mreturn_i_complex_kl_back[:,:,i].flatten()

        sds_o_r_kl_back = np.zeros((n_experiment * m ,  num_delta))
        for i in range(num_delta):
            sds_o_r_kl_back[:,i] = mreturn_o_complex_kl_back[:,:,i].flatten()
            
        plt.plot(deltalist_complex,(np.mean(np.power(np.std(mreturn_i_complex-mreturn_o_complex,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(mreturn_i_complex_wb-mreturn_o_complex_wb,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed', label = "ECMSE-WB")

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(mreturn_i_complex_wb_general-mreturn_o_complex_wb_general,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-WB-Full")

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(mreturn_i_complex_kl_back-mreturn_o_complex_kl_back,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-KL-Back")


        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        
        plt.plot(deltalist_complex,(np.mean(np.power(np.std(mreturn_i_complex-mreturn_o_complex,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2),marker="v",markersize=10,color = '#1f77b4', label = "ECBias^2-KL")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(mreturn_i_complex-mreturn_o_complex,axis=1),2),axis=0),marker="^",markersize=10,color='#1f77b4',label = "ECVar-KL")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        
        plt.plot(deltalist_complex,(np.mean(np.power(np.std(mreturn_i_complex_wb-mreturn_o_complex_wb,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed',color='#ff7f0e',label = "ECMSE-WB")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2),marker="v",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECBias^2-WB")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(mreturn_i_complex_wb-mreturn_o_complex_wb,axis=1),2),axis=0),marker="^",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECVar-WB")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(mreturn_i_complex_wb_general-mreturn_o_complex_wb_general,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#2ca02c',label = "ECMSE-WB-Full")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2),marker="v",markersize=10,color='#2ca02c',linestyle='-.',label = "ECBias^2-WB-Full")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(mreturn_i_complex_wb_general-mreturn_o_complex_wb_general,axis=1),2),axis=0),marker="^",markersize=10,color='#2ca02c',linestyle='-.',label = "ECVar-WB-Full")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        
        plt.plot(deltalist_complex,(np.mean(np.power(np.std(mreturn_i_complex_kl_back-mreturn_o_complex_kl_back,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#B8860B',label = "ECMSE-KL-Back")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2),marker="v",markersize=10,color='#B8860B',linestyle='-.',label = "ECBias^2-KL-Back")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(mreturn_i_complex_kl_back-mreturn_o_complex_kl_back,axis=1),2),axis=0),marker="^",markersize=10,color='#B8860B',linestyle='-.',label = "ECVar-KL-Back")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()

    if param == 'rho':
        
        for k in range(n_experiment):
            print(k)
            data = generate_data()
            mask = missing()
            meanlist,covariancelist = individualposterior(data, mask, mu_p, covp_inv)
    
            deltalist_complex  = np.linspace(0.000, 1.0, num = num_delta)
            rholist_complex = np.linspace(0.001,5.0, num = num_rho)
        
            for i in range(num_rho):
            
                _,final_mu,final_sigma = consensusforwardkl(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],rholist_complex[i])
                completed_data = imputation(data,mask,final_mu,final_sigma)       

                _,final_mu_wb,final_sigma_wb = consensuswasserstein(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],rholist_complex[i],'Vz-l1-norm')
                completed_data_wb = imputation(data,mask,final_mu_wb,final_sigma_wb)
            
                _,final_mu_wb_general,final_sigma_wb_general = consensuswasserstein_general(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],rholist_complex[i],'Vz-l1-norm')
                completed_data_wb_general = imputation(data,mask,final_mu_wb_general,final_sigma_wb_general)
            
                _,final_mu_kl_back,final_sigma_kl_back = consensusbackwardkl(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],np.full(len(meanlist),1/len(meanlist)),10**(-14),20,rholist_complex[i],'Vz-l1-norm')
                completed_data_kl_back = imputation(data,mask,final_mu_kl_back,final_sigma_kl_back)
            
                for j in range(m):
                    _, _,mreturn_i_complex[k,j,i], mreturn_o_complex[k,j,i]  = Greedy(completed_data[j])
                    _, _,mreturn_i_complex_wb[k,j,i], mreturn_o_complex_wb[k,j,i]  = Greedy(completed_data_wb[j])
                    _, _,mreturn_i_complex_wb_general[k,j,i], mreturn_o_complex_wb_general[k,j,i]  = Greedy(completed_data_wb_general[j])
                    _, _,mreturn_i_complex_kl_back[k,j,i], mreturn_o_complex_kl_back[k,j,i]  = Greedy(completed_data_kl_back[j])
                
        sds_r = np.zeros((n_experiment * m ,  num_rho))
        for i in range(num_rho):
            sds_r[:,i] = mreturn_i_complex[:,:,i].flatten()
    
        sds_o_r = np.zeros((n_experiment * m ,  num_rho))
        for i in range(num_rho):
            sds_o_r[:,i] = mreturn_o_complex[:,:,i].flatten()

        sds_r_wb = np.zeros((n_experiment * m ,  num_rho))
        for i in range(num_rho):
            sds_r_wb[:,i] = mreturn_i_complex_wb[:,:,i].flatten()

        sds_o_r_wb = np.zeros((n_experiment * m ,  num_rho))
        for i in range(num_rho):
            sds_o_r_wb[:,i] = mreturn_o_complex_wb[:,:,i].flatten()

        sds_r_wb_general = np.zeros((n_experiment * m ,  num_rho))
        for i in range(num_rho):
            sds_r_wb_general[:,i] = mreturn_i_complex_wb_general[:,:,i].flatten()

        sds_o_r_wb_general = np.zeros((n_experiment * m ,  num_rho))
        for i in range(num_rho):
            sds_o_r_wb_general[:,i] = mreturn_o_complex_wb_general[:,:,i].flatten()
    
        sds_r_kl_back = np.zeros((n_experiment * m ,  num_rho))
        for i in range(num_rho):
            sds_r_kl_back[:,i] = mreturn_i_complex_kl_back[:,:,i].flatten()

        sds_o_r_kl_back = np.zeros((n_experiment * m ,  num_rho))
        for i in range(num_rho):
            sds_o_r_kl_back[:,i] = mreturn_o_complex_kl_back[:,:,i].flatten()


        plt.plot(rholist_complex,(np.mean(np.power(np.std(mreturn_i_complex-mreturn_o_complex,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")

        plt.plot(rholist_complex,(np.mean(np.power(np.std(mreturn_i_complex_wb-mreturn_o_complex_wb,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed', label = "ECMSE-WB")

        plt.plot(rholist_complex,(np.mean(np.power(np.std(mreturn_i_complex_wb_general-mreturn_o_complex_wb_general,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-WB-Full")

        plt.plot(rholist_complex,(np.mean(np.power(np.std(mreturn_i_complex_kl_back-mreturn_o_complex_kl_back,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-KL-Back")


        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        
        plt.plot(rholist_complex,(np.mean(np.power(np.std(mreturn_i_complex-mreturn_o_complex,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2),marker="v",markersize=10,color = '#1f77b4', label = "ECBias^2-KL")
        plt.plot(rholist_complex,np.mean(np.power(np.std(mreturn_i_complex-mreturn_o_complex,axis=1),2),axis=0),marker="^",markersize=10,color='#1f77b4',label = "ECVar-KL")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        
        plt.plot(rholist_complex,(np.mean(np.power(np.std(mreturn_i_complex_wb-mreturn_o_complex_wb,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed',color='#ff7f0e',label = "ECMSE-WB")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2),marker="v",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECBias^2-WB")
        plt.plot(rholist_complex,np.mean(np.power(np.std(mreturn_i_complex_wb-mreturn_o_complex_wb,axis=1),2),axis=0),marker="^",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECVar-WB")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()

        plt.plot(rholist_complex,(np.mean(np.power(np.std(mreturn_i_complex_wb_general-mreturn_o_complex_wb_general,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#2ca02c',label = "ECMSE-WB-Full")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2),marker="v",markersize=10,color='#2ca02c',linestyle='-.',label = "ECBias^2-WB-Full")
        plt.plot(rholist_complex,np.mean(np.power(np.std(mreturn_i_complex_wb_general-mreturn_o_complex_wb_general,axis=1),2),axis=0),marker="^",markersize=10,color='#2ca02c',linestyle='-.',label = "ECVar-WB-Full")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        
        plt.plot(rholist_complex,(np.mean(np.power(np.std(mreturn_i_complex_kl_back-mreturn_o_complex_kl_back,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#B8860B',label = "ECMSE-KL-Back")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2),marker="v",markersize=10,color='#B8860B',linestyle='-.',label = "ECBias^2-KL-Back")
        plt.plot(rholist_complex,np.mean(np.power(np.std(mreturn_i_complex_kl_back-mreturn_o_complex_kl_back,axis=1),2),axis=0),marker="^",markersize=10,color='#B8860B',linestyle='-.',label = "ECVar-KL-Back")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()


# In[ ]:





# In[ ]:




