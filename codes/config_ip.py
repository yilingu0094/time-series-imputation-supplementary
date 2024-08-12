import numpy as np

n = 100  #number of industrial portfolio
T_train = 800 # training period
T_test = 400 #testing period
T_truetest = 560 # out of sample testing period

s = np.loadtxt("/home/yilin/code/bayesian_imputation_code/100_Portfolios_Daily.txt")
fulldata = s[:,1:]
fulldata = 10*fulldata

n_experiment = 50
num_points = 49
m = 100 #number of multiply-imputed dataset
mu_p = np.repeat(0.0,n)
covp_inv = np.diag(np.full(n,0.0))

num_delta = 6
#num_rho = 8
#total_num_rho = num_rho+2
total_num_rho = 7

mreturn_i_complex = np.zeros((n_experiment,m,num_delta))
mreturn_o_complex = np.zeros((n_experiment,m,num_delta))

mreturn_i_complex_wb = np.zeros((n_experiment,m,num_delta))
mreturn_o_complex_wb = np.zeros((n_experiment,m,num_delta))

mreturn_i_complex_wb_general = np.zeros((n_experiment,m,num_delta))
mreturn_o_complex_wb_general = np.zeros((n_experiment,m,num_delta))

mreturn_i_complex_kl_back = np.zeros((n_experiment,m,num_delta))
mreturn_o_complex_kl_back = np.zeros((n_experiment,m,num_delta))

mreturn_i_complex_kl_back_sdr = np.zeros((n_experiment,m,num_delta))
mreturn_o_complex_kl_back_sdr = np.zeros((n_experiment,m,num_delta))



mreturn_i_complex_r = np.zeros((n_experiment,m,total_num_rho))
mreturn_o_complex_r = np.zeros((n_experiment,m,total_num_rho))

mreturn_i_complex_wb_r = np.zeros((n_experiment,m,total_num_rho))
mreturn_o_complex_wb_r = np.zeros((n_experiment,m,total_num_rho))

mreturn_i_complex_wb_general_r = np.zeros((n_experiment,m,total_num_rho))
mreturn_o_complex_wb_general_r = np.zeros((n_experiment,m,total_num_rho))

mreturn_i_complex_kl_back_r = np.zeros((n_experiment,m,total_num_rho))
mreturn_o_complex_kl_back_r = np.zeros((n_experiment,m,total_num_rho))

mreturn_i_complex_kl_back_sdr_r = np.zeros((n_experiment,m,total_num_rho))
mreturn_o_complex_kl_back_sdr_r = np.zeros((n_experiment,m,total_num_rho))

cov = np.cov(np.matrix.transpose(fulldata))

epspath='/home/yilin/code/bayesian_imputation_code/eps_ip/'
