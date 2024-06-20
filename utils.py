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
        print("weight diff:",np.linalg.norm(weight_t-weight_former))
        print("iteration:",iteration)
    
    final_mu = np.zeros(n)
    for i in range(num_posteriors):
        final_mu += weight_t[i]*meanlist[i]
        
    final_sigma = np.zeros((n,n))
    final_sigma_1 = np.zeros((n,n))
    for i in range(num_posteriors):
        final_sigma_1 += weight_t[i]*(np.outer(meanlist[i],meanlist[i])+covariancelist[i])
    final_sigma = final_sigma_1 - np.outer(final_mu,final_mu)
    
    return weight_t, final_mu, final_sigma


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
            #np.multiply(np.sqrt(sv_matrix[i,:]),np.sqrt(sv_matrix[j,:]))是(d_{i1}d_{j1})^{1/2}+...+(d_{in}d_{jn})^{1/2}
            
    #obj = 0
    #for i in range(num_posteriors): 
    #    for j in range(num_posteriors):
    #        obj += weights[i] * weights[j] * np.sum(np.multiply(np.sqrt(sv_matrix[i,:]),np.sqrt(sv_matrix[j,:])))
    
    
    #obj = cp.sum_squares(weights @ sv_matrix)
    obj = cp.sum_squares(sv_matrix.T @ weights.T)
    #obj = cp.quad_form(weights, P)
    delta = delta_r * np.linalg.norm(meanlist[-1] - prediction)#prediction是mu_1
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