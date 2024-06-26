import numpy as np

n = 10 #number of stocks
T_train = 100 # training period
T_test = 100 #testing period
T_truetest = 1000 # out of sample testing period

#parameters for generating normal returns
mu = 0.0 + np.linspace(-0.1, 0.5, n)
cov = np.ones((n,n))+np.eye(n)
cov = 1 * cov

n_experiment = 1
m = 10
mu_p = np.repeat(0.0,n)
covp_inv = np.diag(np.full(n,0.0))

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

epspath='/home/yilin/code/bayesian_imputation_code/eps_syn/'