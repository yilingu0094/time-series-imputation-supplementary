import numpy as np

n = 74  #number of stocks
T_train = 800 # training period
T_test = 400 #testing period
T_truetest = 559 # out of sample testing period

fulldata = np.load('/home/yilin/code/bayesian_imputation_code/74_stock_dataset.npy')

fulldata = 100*fulldata

n_experiment = 50
num_points = 50
m = 100
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

cov = np.cov(np.matrix.transpose(fulldata))

epspath='/home/yilin/code/bayesian_imputation_code/eps_rs/'