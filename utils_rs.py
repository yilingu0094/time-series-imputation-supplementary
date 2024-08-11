import numpy as np
import cvxpy as cp
import scipy
import mosek
from scipy.linalg import sqrtm
import config_rs


def missing(miss_prob=0.5):
    #''True'' represents missing
    mask = np.random.choice([True, False], size = (config_rs.T_train,config_rs.n), p = [miss_prob, 1-miss_prob])#50% probability of missing a value
    mask = np.vstack((mask, np.full((config_rs.T_test + config_rs.T_truetest,config_rs.n),False))) #no missing value for testing period
    return mask#mask is a True, False matrix about missing

def individualposterior(data, mask, mu0, cov0_inv, cov):#the maximum of number of posteriors is T_test+1
    #compute mean and covariance of posterior of \mu given data up to time T in range(...)
    meanlist = []#list of mu_k
    covariancelist = []#list of cov_k
    num_posteriors = config_rs.num_points + 1 #number of posteriors, from T_train to T_train+T_test
    
    covariance_mu_inv = np.copy(cov0_inv)#first input cov0_inv, and then use for loop to sum up the latter part of cov_k
    mean_mu = np.matmul(covariance_mu_inv,mu0)#matrix multiplication of covariance_mu_inv and mu0
    
    for t in range(config_rs.T_train + config_rs.T_test):
        
        if not np.all(mask[t,:] == np.full(config_rs.n,True)):#except for all missing values at the t-th sample, otherwise run the following
            ind = (mask[t,:] == np.full(config_rs.n,False))#judge each element of the t-th sample missing or not
            cov_t = config_rs.cov[np.ix_(ind,ind)]#cov_t = cov[non-missing entries], i.e. Omega_Xt
            cov_inv_expand = np.zeros((config_rs.n,config_rs.n))
            cov_inv_expand[np.ix_(ind,ind)] = scipy.linalg.pinv(cov_t)#(P_Mt)(Omega_Xt)

            covariance_mu_inv += cov_inv_expand#sum_{Tk}((P_Mt)^{-1}(Omega_Xt))

            mean_t = np.zeros(config_rs.n)
            mean_t[ind] = data[t,ind]#mean_t = ((P_Mt)^{-1}(Xt))
            mean_mu += np.matmul(cov_inv_expand,mean_t)#sum_{Tk}((P_Mt)^{-1}(Omega_Xt))((P_Mt)^{-1}(Xt))
        
        if t >= (config_rs.T_train - 1):#when t>= T_train-1, each for-loop generates a new (cov_k,mu_k). Put (cov_k,mu_k) into list, and we have a list of covariance and mean of posteriors.
            if (t - config_rs.T_train + 1) % (config_rs.T_test/num_posteriors) == 0:
                covariance_mu = scipy.linalg.pinv(covariance_mu_inv)#cov_k
                covariancelist.append(covariance_mu)
                meanlist.append(np.matmul(covariance_mu, mean_mu))#meanlist append mu_k
        
    return meanlist,covariancelist  

def consensusforwardkl(meanlist,covariancelist,delta_r,prediction,rho):
    
    num_posteriors = len(meanlist)#number of posteriors
    config_rs.n = len(meanlist[0])#number of stocks
    # Define optimization variables
    weights = cp.Variable(num_posteriors)
    gamma = cp.Variable(config_rs.n)
    # v, s, vt = np.linalg.svd(covariancelist[0].values)
    
    s, v = np.linalg.eigh(covariancelist[0])#s is eigenvalue, and v is eigenvector which is the orthogonal matrix V
    
    s_list = []
    
    # s_list.append(s)
    
   
    
    # s, v = np.linalg.eigh(covariancelist[1].values)
    # s_list.append(s)
    
    # for i in np.arange(1,num_posteriors,1):
    for i in range(num_posteriors):
        s_temp = np.zeros(config_rs.n)
        for j in range(config_rs.n):
            s_temp[j] = np.inner(v[:,j],np.matmul(covariancelist[i],v[:,j]))#dkj, which follows from the projection on Nv
        s_list.append(s_temp)#s_temp is dk, dk is the diagonal of Dk, and s_list store all the dk
        
    c = np.zeros((num_posteriors,config_rs.n))#ckj stores in c
    
    for i in range(num_posteriors):
        for j in range(config_rs.n):
            c[i, j] = np.inner(v[:,j],meanlist[i])/s_list[i][j]#ckj=vj^{T}mu_k/dkj
            
        
    sv_matrix = np.array(s_list)#store the matrix of dk     
    inverse_sv = 1.0 / sv_matrix
    
    #objective_fun = [cp.power(cp.sum(cp.multiply(inverse_sv[:,j], weights)),-1) for j in range(config_rs.n)]
    
    obj = cp.sum([gamma[j] for j in range(config_rs.n)])#sum(gamma_j)
    
    # Run optimization
    objective = cp.Minimize(obj)
    delta = delta_r * max([np.abs(c[-1,j]/inverse_sv[-1,j] - v[:,j].dot(prediction)) for j in range(config_rs.n)])
    constraints = [weights >= 0,
                   cp.sum(weights) == 1]
    for j in range(config_rs.n):
        constraints.append(cp.sum(cp.multiply(c[:,j],weights)) <= (delta/rho + v[:,j].dot(prediction)) * cp.sum(cp.multiply(inverse_sv[:,j], weights)))
        constraints.append(cp.sum(cp.multiply(c[:,j],weights)) >= (-delta/rho + v[:, j].dot(prediction)) * cp.sum(cp.multiply(inverse_sv[:,j], weights)))
        #constraints.append(4 +  cp.power(cp.sum(cp.multiply(inverse_sv[:,j], weights))-gamma[j],2)<= cp.power(cp.sum(cp.multiply(inverse_sv[:,j], weights))+gamma[j],2))   
        A = np.zeros((2,num_posteriors))
        B = np.zeros((2,config_rs.n))
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
    
    #calculate the covariance and mean of the barycenter
    final_sigma = scipy.linalg.inv(sum([solution[i] * scipy.linalg.inv(covariancelist[i]) for i in range(num_posteriors)]))
    final_mu = final_sigma.dot(sum([solution[i] * np.inner(scipy.linalg.inv(covariancelist[i]), meanlist[i]) for i in range(num_posteriors)]))
    
    return solution, final_mu, final_sigma

#def consensuswasserstein(meanlist,covariancelist,delta_r,prediction,rho,set_choice):
    
#    num_posteriors = len(meanlist)
#    config_rs.n = len(meanlist[0])
    
#    weights = cp.Variable(2)
#    s, v = np.linalg.eigh(covariancelist[0])#v is the orthogonal matrix
    
        
#    Sigma1 = covariancelist[0]
#    Sigma2 = covariancelist[-1]
#    temp = sqrtm(Sigma2) @ Sigma1 @ sqrtm(Sigma2)
#   Psi = sqrtm(Sigma2) @ np.real(scipy.linalg.inv(sqrtm(temp))) @ sqrtm(Sigma2)
    
#    P = np.zeros((2,2))
#    P[0,0] = np.trace(Sigma1)
#    P[1,1] = np.trace(Sigma2)
#    P[0,1] = np.trace(Sigma1 @ Psi)
#    P[1,0] = np.trace(Sigma1 @ Psi)
   
#    obj = cp.quad_form(weights, P)
    
#    delta = delta_r * np.linalg.norm(meanlist[-1] - prediction)
#    constraints = [weights >= 0,
#                   cp.sum(weights) == 1]
    
#    if set_choice == 'l2-norm':
#        constraints.append(cp.norm(weights[0]*meanlist[0]+weights[1]*meanlist[1]-prediction)<=delta/rho)
#    if set_choice == 'Vz-l1-norm':
#        constraints.append(cp.norm(v @ (weights[0]*meanlist[0]+weights[1]*meanlist[1]-prediction),"inf") <= delta/rho)
        
    
#    prob = cp.Problem(cp.Minimize(obj), constraints)
#    prob.solve(solver=cp.MOSEK, verbose=True)
    
#    solution = weights.value
   
#    final_mu = meanlist[0] * solution[0] + meanlist[-1] * solution[1]
#    final_sigma = (solution[0]*np.eye(config_rs.n) + solution[1] * Psi) @ Sigma1 @ (solution[0]*np.eye(config_rs.n) + solution[1] * Psi)
    
#    return solution, final_mu, final_sigma

def consensuswasserstein_general(meanlist,covariancelist,delta_r, prediction, rho, set_choice):
    
    num_posteriors = len(meanlist)
    config_rs.n = len(meanlist[0])
    
    # Define optimization variables
    weights = cp.Variable(num_posteriors)
    
    # v, s, vt = np.linalg.svd(covariancelist[0].values)
    
    s, v = np.linalg.eigh(covariancelist[0])#v is the orthogonal matrix V
    
    s_list = []
    
    # for i in np.arange(1,num_posteriors,1):
    for i in range(num_posteriors):
        s_temp = np.zeros(config_rs.n)
        for j in range(config_rs.n):
            s_temp[j] = np.inner(v[:,j],np.matmul(covariancelist[i],v[:,j]))
        s_list.append(s_temp)#s_temp is dk, dk is the diagonal of Dk, and s_list stores all the dk.
    
    sv_matrix = np.array(s_list)
    
    P = np.zeros((num_posteriors,num_posteriors))
    reg_term = 1e-6 * np.eye(num_posteriors)
    for i in range(num_posteriors): 
        for j in range(num_posteriors):
            P[i,j] = np.sum(np.multiply(np.sqrt(sv_matrix[i,:]),np.sqrt(sv_matrix[j,:])))#P is the Gram matrix
    P += reg_term#regularization term used to avoid floating-point arithmetic issues.

    obj = cp.quad_form(weights, P)
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
    
    
    final_mu = np.zeros(config_rs.n)
    for i in range(num_posteriors):
        final_mu += solution[i]*meanlist[i]
    final_sigma = np.zeros((config_rs.n,config_rs.n))
    for i in range(num_posteriors):
        for j in range(config_rs.n):
            final_sigma += solution[i] * np.sqrt(sv_matrix[i,j]) * np.outer(v[:,j],v[:,j])
    final_sigma = final_sigma @ final_sigma
    
    return solution, final_mu, final_sigma

def consensusbackwardkl(meanlist,covariancelist,delta_r,prediction,weight_init,epsilon,max_iter,rho,set_choice):#weight_init = np.full(num_posteriors,1/num_posteriors)
    
    num_posteriors = len(meanlist)
    config_rs.n = len(meanlist[0])
    
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
        
        obj = obj1+obj2+obj3#the linearized objective for DC programming
    
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
        #for j in range(config_rs.n): #j:0,...config_rs.n-1
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
    
    final_mu = np.zeros(config_rs.n)
    for i in range(num_posteriors):
        final_mu += weight_t[i]*meanlist[i]
        
    final_sigma = np.zeros((config_rs.n,config_rs.n))
    final_sigma_1 = np.zeros((config_rs.n,config_rs.n))
    for i in range(num_posteriors):
        final_sigma_1 += weight_t[i]*(np.outer(meanlist[i],meanlist[i])+covariancelist[i])
    final_sigma = final_sigma_1 - np.outer(final_mu,final_mu)
    
    return weight_t, final_mu, final_sigma

#def consensusbackwardkl_sdr(meanlist,covariancelist,delta_r,prediction,rho,set_choice):

#    num_posteriors = len(meanlist)
#    config_rs.n = len(meanlist[0])
    
    # Define optimization variables
#    weights = cp.Variable((num_posteriors,1))
#    Lambda_w = cp.Variable((num_posteriors,num_posteriors),symmetric=True)
    
#    s, v = np.linalg.eigh(covariancelist[0])

#    obj1 = 0
#    for i in range(num_posteriors):
#        obj1 += weights[i]*cp.trace(np.outer(meanlist[i],meanlist[i])+covariancelist[i])

#    M_matrix = np.zeros((num_posteriors,num_posteriors))
#    for i in range(num_posteriors): 
#        for j in range(num_posteriors):
#            M_matrix[i,j] = np.inner(meanlist[i],meanlist[j])

#    obj = obj1-cp.trace(Lambda_w @ M_matrix)

#    delta = delta_r * np.linalg.norm(meanlist[-1] - prediction)
#    #reg_term = 1e-6 * np.eye(num_posteriors)
#    constraints = [weights >= 0,
#                   cp.sum(weights) == 1,
#                   cp.bmat([[Lambda_w,weights],[weights.T,np.ones((1,1))]]) >> 0,
#                   cp.norm(Lambda_w,"nuc") <= 1]
    
#    temp = 0
#    for i in range(num_posteriors):
#        temp += weights[i]*meanlist[i]#\sum(\lambda_k\mu_k)
#    if set_choice == 'l1-norm': 
#        constraints.append(cp.norm(temp-prediction,"inf") <= delta/rho)
#    if set_choice == 'Vz-l1-norm':
#        constraints.append(cp.norm(v @ (temp-prediction),"inf") <= delta/rho)

#    prob = cp.Problem(cp.Minimize(obj), constraints)
#    prob.solve(solver=cp.MOSEK)#, verbose=True
#    solution = weights.value
#    final_mu = np.zeros(config_rs.n)
#    for i in range(num_posteriors):
#        final_mu += solution[i]*meanlist[i]
#    final_sigma = np.zeros((config_rs.n,config_rs.n))
#    final_sigma_1 = np.zeros((config_rs.n,config_rs.n))
#    for i in range(num_posteriors):
#        final_sigma_1 += solution[i]*(np.outer(meanlist[i],meanlist[i])+covariancelist[i])
#    final_sigma = final_sigma_1 - np.outer(final_mu,final_mu)    
    
#    return solution, final_mu, final_sigma

def imputation(data, mask, final_mu,final_sigma):  
    total_time,num_stocks = data.shape
    completed_data = np.zeros((config_rs.m,total_time,num_stocks))
    for k in range(config_rs.m):
        data_copy = np.copy(data)
        unconditionalmean = np.random.multivariate_normal(final_mu, final_sigma)
        for t in range(config_rs.T_train):
            if np.all(mask[t,:] == np.full(config_rs.n,True)):
                data_copy[t,:] = unconditionalmean
            elif np.all(mask[t,:] == np.full(config_rs.n,False)):
                pass
            else:
                ind_miss = (mask[t,:] == np.full(config_rs.n,True))
                ind_obs = (mask[t,:] == np.full(config_rs.n,False))
                len_miss = len(data_copy[t,ind_miss])
                data_copy[t,ind_miss] = np.zeros(len_miss)
                cov11 = config_rs.cov[np.ix_(ind_miss,ind_obs)]
                cov12_inv = np.linalg.inv(config_rs.cov[np.ix_(ind_obs,ind_obs)])
                missing_condi_mean = unconditionalmean[ind_miss] + np.matmul(np.matmul(cov11,cov12_inv),data_copy[t,ind_obs] - unconditionalmean[ind_obs])
                missing_condi_cov = config_rs.cov[np.ix_(ind_miss,ind_miss)] - np.matmul(cov11,np.matmul(cov12_inv,np.matrix.transpose(cov11)))
                data_copy[t,ind_miss] = missing_condi_mean
        completed_data[k,:,:] = data_copy
    return completed_data

def Greedy(data1):
    data = np.copy(data1)
    data = data/100
    mean = np.mean(np.exp(data[:config_rs.T_train,:])-1.0,axis = 0)
    #if np.sum(mean) <= 0:
    #    print('whoops')
    weights = mean / np.linalg.norm(mean,2)
    #c_returns = 1.0
    returns = np.zeros(config_rs.T_test)
    for t in np.arange(config_rs.T_train,config_rs.T_train+config_rs.T_test,1):
        returns[t-config_rs.T_train] = np.inner(np.exp(data[t,:])-1.0, weights)
        #c_returns *= 1.0 + returns[t-config_rs.T_train]
    sharper = np.mean(returns)/np.std(returns)
    
    #o_returns = 1.0
    returns_o = np.zeros(config_rs.T_truetest)
    for t in np.arange(config_rs.T_train+config_rs.T_test,config_rs.T_train+config_rs.T_test+config_rs.T_truetest,1):
        returns_o[t-config_rs.T_train - config_rs.T_test] = np.inner(np.exp(data[t,:])-1.0, weights)
       # o_returns *= 1.0 + returns_o[t-config_rs.T_train - config_rs.T_test]
    o_sharper = np.mean(returns_o)/np.std(returns_o)

    return sharper, o_sharper,np.mean(returns),np.mean(returns_o)
    #return np.mean(returns),np.mean(returns_o)



