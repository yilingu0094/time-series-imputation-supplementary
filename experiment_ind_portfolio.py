import numpy as np
import cvxpy as cp
import scipy
import mosek
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

import utils_ip
import config_ip

def general_experiment(param, hyperparam):
    #param is the testing parameter at experiment: delta or rho
    #hyperparam is the hyperparamter at experiment: e.g. when testing delta, the hyperparameter rho that we use.
    
    if param == 'delta':

        for k in range(config_ip.n_experiment):
            data = config_ip.fulldata
            mask = utils_ip.missing()
            meanlist,covariancelist = utils_ip.individualposterior(data, mask, config_ip.mu_p, config_ip.covp_inv, config_ip.cov)
    
            deltalist_complex  = np.linspace(0.0, 1.0, num = config_ip.num_delta)
            rholist_complex = np.array([1e-4, 1e-2, 1e-1, 1, 10, 100, 1e4])#rho should not equal to 0.
        
            for i in range(config_ip.num_delta):
        
                _,final_mu,final_sigma = utils_ip.consensusforwardkl(meanlist,covariancelist,deltalist_complex[i],meanlist[0],rholist_complex[hyperparam])
                completed_data = utils_ip.imputation(data,mask,final_mu,final_sigma)
        
                #_,final_mu_wb,final_sigma_wb = utils_ip.consensuswasserstein(meanlist,covariancelist,deltalist_complex[i],meanlist[0],rholist_complex[hyperparam],'Vz-l1-norm')
                #completed_data_wb = utils_ip.imputation(data,mask,final_mu_wb,final_sigma_wb)
        
                _,final_mu_wb_general,final_sigma_wb_general = utils_ip.consensuswasserstein_general(meanlist,covariancelist,deltalist_complex[i],meanlist[0],rholist_complex[hyperparam],'Vz-l1-norm')
                completed_data_wb_general = utils_ip.imputation(data,mask,final_mu_wb_general,final_sigma_wb_general)
        
                _,final_mu_kl_back,final_sigma_kl_back = utils_ip.consensusbackwardkl(meanlist,covariancelist,deltalist_complex[i],meanlist[0],np.full(len(meanlist),1/len(meanlist)),10**(-15),20,rholist_complex[hyperparam],'Vz-l1-norm')
                completed_data_kl_back = utils_ip.imputation(data,mask,final_mu_kl_back,final_sigma_kl_back)

                #_,final_mu_kl_back,final_sigma_kl_back = utils_ip.consensusbackwardkl_sdr(meanlist,covariancelist,deltalist_complex[i],meanlist[0],rholist_complex[hyperparam],'Vz-l1-norm')
                #completed_data_kl_back_sdr = utils_ip.imputation(data,mask,final_mu_kl_back,final_sigma_kl_back)

        
                for j in range(config_ip.m):
                    _, _,config_ip.mreturn_i_complex[k,j,i], config_ip.mreturn_o_complex[k,j,i]  = utils_ip.Greedy(completed_data[j])
                    #_, _,config_ip.mreturn_i_complex_wb[k,j,i], config_ip.mreturn_o_complex_wb[k,j,i]  = utils_ip.Greedy(completed_data_wb[j])
                    _, _,config_ip.mreturn_i_complex_wb_general[k,j,i], config_ip.mreturn_o_complex_wb_general[k,j,i]  = utils_ip.Greedy(completed_data_wb_general[j])
                    _, _,config_ip.mreturn_i_complex_kl_back[k,j,i], config_ip.mreturn_o_complex_kl_back[k,j,i]  = utils_ip.Greedy(completed_data_kl_back[j])
                    #_, _,config_ip.mreturn_i_complex_kl_back_sdr[k,j,i], config_ip.mreturn_o_complex_kl_back_sdr[k,j,i]  = utils_ip.Greedy(completed_data_kl_back_sdr[j])
                
        sds_r = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_r[:,i] = config_ip.mreturn_i_complex[:,:,i].flatten()
    
        sds_o_r = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_o_r[:,i] = config_ip.mreturn_o_complex[:,:,i].flatten()

        #sds_r_wb = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        #for i in range(config_ip.num_delta):
            #sds_r_wb[:,i] = config_ip.mreturn_i_complex_wb[:,:,i].flatten()

        #sds_o_r_wb = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        #for i in range(config_ip.num_delta):
            #sds_o_r_wb[:,i] = config_ip.mreturn_o_complex_wb[:,:,i].flatten()

        sds_r_wb_general = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_r_wb_general[:,i] = config_ip.mreturn_i_complex_wb_general[:,:,i].flatten()

        sds_o_r_wb_general = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_o_r_wb_general[:,i] = config_ip.mreturn_o_complex_wb_general[:,:,i].flatten()
    
        sds_r_kl_back = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_r_kl_back[:,i] = config_ip.mreturn_i_complex_kl_back[:,:,i].flatten()

        sds_o_r_kl_back = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_o_r_kl_back[:,i] = config_ip.mreturn_o_complex_kl_back[:,:,i].flatten()

        #sds_r_kl_back_sdr = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        #for i in range(config_ip.num_delta):
            #sds_r_kl_back_sdr[:,i] = config_ip.mreturn_i_complex_kl_back_sdr[:,:,i].flatten()
        
        #sds_o_r_kl_back_sdr = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        #for i in range(config_ip.num_delta):
            #sds_o_r_kl_back_sdr[:,i] = config_ip.mreturn_o_complex_kl_back_sdr[:,:,i].flatten()


        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex-config_ip.mreturn_o_complex,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")

        #plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb-config_ip.mreturn_o_complex_wb,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed', label = "ECMSE-WB")

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general-config_ip.mreturn_o_complex_wb_general,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-WB-Full")

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back-config_ip.mreturn_o_complex_kl_back,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-KL-Back")

        #plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back_sdr-config_ip.mreturn_o_complex_kl_back_sdr,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back_sdr - sds_o_r_kl_back_sdr,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-KL-Back-SDR")



        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_rho={}_g1.png".format(param,rholist_complex[hyperparam]),format='png')
        plt.clf()
        
        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex-config_ip.mreturn_o_complex,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2),marker="v",markersize=10,color = '#1f77b4', label = "ECBias^2-KL")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex-config_ip.mreturn_o_complex,axis=1),2),axis=0),marker="^",markersize=10,color='#1f77b4',label = "ECVar-KL")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_rho={}_g2.png".format(param,rholist_complex[hyperparam]),format='png')
        plt.clf()
        
        #plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb-config_ip.mreturn_o_complex_wb,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed',color='#ff7f0e',label = "ECMSE-WB")
        #plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2),marker="v",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECBias^2-WB")
        #plt.plot(deltalist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb-config_ip.mreturn_o_complex_wb,axis=1),2),axis=0),marker="^",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECVar-WB")
        #plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        #plt.legend()
        #plt.legend(fontsize=15)
        #plt.show()
        #plt.savefig(config_ip.epspath + "/{}_rho={}_g3.png".format(param,rholist_complex[hyperparam]),format='png')
        #plt.clf()

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general-config_ip.mreturn_o_complex_wb_general,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#2ca02c',label = "ECMSE-WB-Full")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2),marker="v",markersize=10,color='#2ca02c',linestyle='-.',label = "ECBias^2-WB-Full")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general-config_ip.mreturn_o_complex_wb_general,axis=1),2),axis=0),marker="^",markersize=10,color='#2ca02c',linestyle='-.',label = "ECVar-WB-Full")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_rho={}_g3.png".format(param,rholist_complex[hyperparam]),format='png')
        plt.clf()
        
        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back-config_ip.mreturn_o_complex_kl_back,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#B8860B',label = "ECMSE-KL-Back")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2),marker="v",markersize=10,color='#B8860B',linestyle='-.',label = "ECBias^2-KL-Back")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back-config_ip.mreturn_o_complex_kl_back,axis=1),2),axis=0),marker="^",markersize=10,color='#B8860B',linestyle='-.',label = "ECVar-KL-Back")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_rho={}_g4.png".format(param,rholist_complex[hyperparam]),format='png')
        plt.clf()

        #plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back_sdr-config_ip.mreturn_o_complex_kl_back_sdr,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back_sdr - sds_o_r_kl_back_sdr,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#B8860B',label = "ECMSE-KL-Back-SDR")
        #plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_kl_back_sdr - sds_o_r_kl_back_sdr,axis=0),0),2),marker="v",markersize=10,color='#B8860B',linestyle='-.',label = "ECBias^2-KL-Back-SDR")
        #plt.plot(deltalist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back_sdr-config_ip.mreturn_o_complex_kl_back_sdr,axis=1),2),axis=0),marker="^",markersize=10,color='#B8860B',linestyle='-.',label = "ECVar-KL-Back-SDR")
        #plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        #plt.legend()
        #plt.legend(fontsize=15)
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.show()
        #plt.savefig(config_ip.epspath + "/{}_rho={}_g5.png".format(param,rholist_complex[hyperparam]),format='png')
        #plt.clf()    

    if param == 'rho':
        
        for k in range(config_ip.n_experiment):
            print(k)
            data = config_ip.fulldata
            mask = utils_ip.missing()
            meanlist,covariancelist = utils_ip.individualposterior(data, mask, config_ip.mu_p, config_ip.covp_inv, config_ip.cov)
    
            deltalist_complex  = np.linspace(0.0, 1.0, num = config_ip.num_delta)
            rholist_complex = np.array([1e-4, 1e-2, 1e-1, 1, 10, 100, 1e4])#rho should not equal to 0.
        
            for i in range(config_ip.total_num_rho):
            
                _,final_mu,final_sigma = utils_ip.consensusforwardkl(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],rholist_complex[i])
                completed_data = utils_ip.imputation(data,mask,final_mu,final_sigma)       

                #_,final_mu_wb,final_sigma_wb = utils_ip.consensuswasserstein(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],rholist_complex[i],'Vz-l1-norm')
                #completed_data_wb = utils_ip.imputation(data,mask,final_mu_wb,final_sigma_wb)
            
                _,final_mu_wb_general,final_sigma_wb_general = utils_ip.consensuswasserstein_general(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],rholist_complex[i],'Vz-l1-norm')
                completed_data_wb_general = utils_ip.imputation(data,mask,final_mu_wb_general,final_sigma_wb_general)
            
                _,final_mu_kl_back,final_sigma_kl_back = utils_ip.consensusbackwardkl(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],np.full(len(meanlist),1/len(meanlist)),10**(-15),20,rholist_complex[i],'Vz-l1-norm')
                completed_data_kl_back = utils_ip.imputation(data,mask,final_mu_kl_back,final_sigma_kl_back)

                #_,final_mu_kl_back,final_sigma_kl_back = utils_ip.consensusbackwardkl_sdr(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],rholist_complex[i],'Vz-l1-norm')
                #completed_data_kl_back_sdr = utils_ip.imputation(data,mask,final_mu_kl_back,final_sigma_kl_back)
            
                for j in range(config_ip.m):
                    _, _,config_ip.mreturn_i_complex_r[k,j,i], config_ip.mreturn_o_complex_r[k,j,i]  = utils_ip.Greedy(completed_data[j])
                    #_, _,config_ip.mreturn_i_complex_wb_r[k,j,i], config_ip.mreturn_o_complex_wb_r[k,j,i]  = utils_ip.Greedy(completed_data_wb[j])
                    _, _,config_ip.mreturn_i_complex_wb_general_r[k,j,i], config_ip.mreturn_o_complex_wb_general_r[k,j,i]  = utils_ip.Greedy(completed_data_wb_general[j])
                    _, _,config_ip.mreturn_i_complex_kl_back_r[k,j,i], config_ip.mreturn_o_complex_kl_back_r[k,j,i]  = utils_ip.Greedy(completed_data_kl_back[j])
                    #_, _,config_ip.mreturn_i_complex_kl_back_sdr_r[k,j,i], config_ip.mreturn_o_complex_kl_back_sdr_r[k,j,i]  = utils_ip.Greedy(completed_data_kl_back_sdr[j])                

        sds_r = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.total_num_rho))
        for i in range(config_ip.total_num_rho):
            sds_r[:,i] = config_ip.mreturn_i_complex_r[:,:,i].flatten()
    
        sds_o_r = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.total_num_rho))
        for i in range(config_ip.total_num_rho):
            sds_o_r[:,i] = config_ip.mreturn_o_complex_r[:,:,i].flatten()

        #sds_r_wb = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.total_num_rho))
        #for i in range(config_ip.total_num_rho):
            #sds_r_wb[:,i] = config_ip.mreturn_i_complex_wb_r[:,:,i].flatten()

        #sds_o_r_wb = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.total_num_rho))
        #for i in range(config_ip.total_num_rho):
            #sds_o_r_wb[:,i] = config_ip.mreturn_o_complex_wb_r[:,:,i].flatten()

        sds_r_wb_general = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.total_num_rho))
        for i in range(config_ip.total_num_rho):
            sds_r_wb_general[:,i] = config_ip.mreturn_i_complex_wb_general_r[:,:,i].flatten()

        sds_o_r_wb_general = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.total_num_rho))
        for i in range(config_ip.total_num_rho):
            sds_o_r_wb_general[:,i] = config_ip.mreturn_o_complex_wb_general_r[:,:,i].flatten()
    
        sds_r_kl_back = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.total_num_rho))
        for i in range(config_ip.total_num_rho):
            sds_r_kl_back[:,i] = config_ip.mreturn_i_complex_kl_back_r[:,:,i].flatten()

        sds_o_r_kl_back = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.total_num_rho))
        for i in range(config_ip.total_num_rho):
            sds_o_r_kl_back[:,i] = config_ip.mreturn_o_complex_kl_back_r[:,:,i].flatten()

        #sds_r_kl_back_sdr = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.total_num_rho))
        #for i in range(config_ip.total_num_rho):
            #sds_r_kl_back_sdr[:,i] = config_ip.mreturn_i_complex_kl_back_sdr_r[:,:,i].flatten()
        
        #sds_o_r_kl_back_sdr = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.total_num_rho))
        #for i in range(config_ip.total_num_rho):
            #sds_o_r_kl_back_sdr[:,i] = config_ip.mreturn_o_complex_kl_back_sdr_r[:,:,i].flatten()

        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_r-config_ip.mreturn_o_complex_r,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")

        #plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_r-config_ip.mreturn_o_complex_wb_r,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed', label = "ECMSE-WB")

        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general_r-config_ip.mreturn_o_complex_wb_general_r,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-WB-Full")

        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back_r-config_ip.mreturn_o_complex_kl_back_r,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-KL-Back")

        #plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back_sdr_r-config_ip.mreturn_o_complex_kl_back_sdr_r,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back_sdr - sds_o_r_kl_back_sdr,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-KL-Back-SDR")



        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_delta={}_g1.png".format(param,deltalist_complex[hyperparam]),format='png')
        plt.clf()
        
        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_r-config_ip.mreturn_o_complex_r,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2),marker="v",markersize=10,color = '#1f77b4', label = "ECBias^2-KL")
        plt.plot(rholist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_r-config_ip.mreturn_o_complex_r,axis=1),2),axis=0),marker="^",markersize=10,color='#1f77b4',label = "ECVar-KL")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_delta={}_g2.png".format(param,deltalist_complex[hyperparam]),format='png')
        plt.clf()
        
        #plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_r-config_ip.mreturn_o_complex_wb_r,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed',color='#ff7f0e',label = "ECMSE-WB")
        #plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2),marker="v",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECBias^2-WB")
        #plt.plot(rholist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_r-config_ip.mreturn_o_complex_wb_r,axis=1),2),axis=0),marker="^",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECVar-WB")
        #plt.xlabel(r'$\rho$',fontsize=15)
        #plt.legend()
        #plt.legend(fontsize=15)
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.show()
        #plt.savefig(config_ip.epspath + "/{}_delta={}_g3.png".format(param,deltalist_complex[hyperparam]),format='png')
        #plt.clf()

        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general_r-config_ip.mreturn_o_complex_wb_general_r,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#2ca02c',label = "ECMSE-WB-Full")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2),marker="v",markersize=10,color='#2ca02c',linestyle='-.',label = "ECBias^2-WB-Full")
        plt.plot(rholist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general_r-config_ip.mreturn_o_complex_wb_general_r,axis=1),2),axis=0),marker="^",markersize=10,color='#2ca02c',linestyle='-.',label = "ECVar-WB-Full")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_delta={}_g3.png".format(param,deltalist_complex[hyperparam]),format='png')
        plt.clf()
        
        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back_r-config_ip.mreturn_o_complex_kl_back_r,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#B8860B',label = "ECMSE-KL-Back")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2),marker="v",markersize=10,color='#B8860B',linestyle='-.',label = "ECBias^2-KL-Back")
        plt.plot(rholist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back_r-config_ip.mreturn_o_complex_kl_back_r,axis=1),2),axis=0),marker="^",markersize=10,color='#B8860B',linestyle='-.',label = "ECVar-KL-Back")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_delta={}_g4.png".format(param,deltalist_complex[hyperparam]),format='png')
        plt.clf()

        #plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back_sdr_r-config_ip.mreturn_o_complex_kl_back_sdr_r,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back_sdr - sds_o_r_kl_back_sdr,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#B8860B',label = "ECMSE-KL-Back-SDR")
        #plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_kl_back_sdr - sds_o_r_kl_back_sdr,axis=0),0),2),marker="v",markersize=10,color='#B8860B',linestyle='-.',label = "ECBias^2-KL-Back-SDR")
        #plt.plot(rholist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back_sdr_r-config_ip.mreturn_o_complex_kl_back_sdr_r,axis=1),2),axis=0),marker="^",markersize=10,color='#B8860B',linestyle='-.',label = "ECVar-KL-Back-SDR")
        #plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        #plt.legend()
        #plt.legend(fontsize=15)
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.show()
        #plt.savefig(config_ip.epspath + "/{}_delta={}_g5.png".format(param,deltalist_complex[hyperparam]),format='png')
        #plt.clf()

def specific_experiment(param, hyperparam):
    #param is the testing parameter at experiment: delta or rho
    #hyperparam is the hyperparamter at experiment: e.g. when testing delta, the hyperparameter rho that we use.
    
    if param == 'delta':

        for k in range(config_ip.n_experiment):
            print(k)
            data = config_ip.fulldata
            mask = utils_ip.missing()
            meanlist,covariancelist = utils_ip.individualposterior(data, mask, config_ip.mu_p, config_ip.covp_inv, config_ip.cov)
    
            deltalist_complex  = np.linspace(0.000, 1.0, num = config_ip.num_delta)
            rholist_complex = np.linspace(0.001,5.0, num = config_ip.num_rho)#rho should not equal to 0.
        
            for i in range(config_ip.num_delta):
        
                _,final_mu,final_sigma = utils_ip.consensusforwardkl(meanlist,covariancelist,deltalist_complex[i],meanlist[0],rholist_complex[hyperparam])
                completed_data = utils_ip.imputation(data,mask,final_mu,final_sigma)
        
                _,final_mu_wb,final_sigma_wb = utils_ip.consensuswasserstein(meanlist,covariancelist,deltalist_complex[i],meanlist[0],rholist_complex[hyperparam],'l2-norm')
                completed_data_wb = utils_ip.imputation(data,mask,final_mu_wb,final_sigma_wb)
        
                _,final_mu_wb_general,final_sigma_wb_general = utils_ip.consensuswasserstein_general(meanlist,covariancelist,deltalist_complex[i],meanlist[0],rholist_complex[hyperparam],'l2-norm')
                completed_data_wb_general = utils_ip.imputation(data,mask,final_mu_wb_general,final_sigma_wb_general)
        
                _,final_mu_kl_back,final_sigma_kl_back = utils_ip.consensusbackwardkl(meanlist,covariancelist,deltalist_complex[i],meanlist[0],np.full(len(meanlist),1/len(meanlist)),10**(-15),50,rholist_complex[hyperparam],'l1-norm')
                completed_data_kl_back = utils_ip.imputation(data,mask,final_mu_kl_back,final_sigma_kl_back)
        
                for j in range(config_ip.m):
                    _, _,config_ip.mreturn_i_complex[k,j,i], config_ip.mreturn_o_complex[k,j,i]  = utils_ip.Greedy(completed_data[j])
                    _, _,config_ip.mreturn_i_complex_wb[k,j,i], config_ip.mreturn_o_complex_wb[k,j,i]  = utils_ip.Greedy(completed_data_wb[j])
                    _, _,config_ip.mreturn_i_complex_wb_general[k,j,i], config_ip.mreturn_o_complex_wb_general[k,j,i]  = utils_ip.Greedy(completed_data_wb_general[j])
                    _, _,config_ip.mreturn_i_complex_kl_back[k,j,i], config_ip.mreturn_o_complex_kl_back[k,j,i]  = utils_ip.Greedy(completed_data_kl_back[j])
                
        sds_r = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_r[:,i] = config_ip.mreturn_i_complex[:,:,i].flatten()
    
        sds_o_r = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_o_r[:,i] = config_ip.mreturn_o_complex[:,:,i].flatten()

        sds_r_wb = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_r_wb[:,i] = config_ip.mreturn_i_complex_wb[:,:,i].flatten()

        sds_o_r_wb = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_o_r_wb[:,i] = config_ip.mreturn_o_complex_wb[:,:,i].flatten()

        sds_r_wb_general = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_r_wb_general[:,i] = config_ip.mreturn_i_complex_wb_general[:,:,i].flatten()

        sds_o_r_wb_general = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_o_r_wb_general[:,i] = config_ip.mreturn_o_complex_wb_general[:,:,i].flatten()
    
        sds_r_kl_back = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_r_kl_back[:,i] = config_ip.mreturn_i_complex_kl_back[:,:,i].flatten()

        sds_o_r_kl_back = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_delta))
        for i in range(config_ip.num_delta):
            sds_o_r_kl_back[:,i] = config_ip.mreturn_o_complex_kl_back[:,:,i].flatten()
            
        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex-config_ip.mreturn_o_complex,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb-config_ip.mreturn_o_complex_wb,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed', label = "ECMSE-WB")

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general-config_ip.mreturn_o_complex_wb_general,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-WB-Full")

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back-config_ip.mreturn_o_complex_kl_back,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-KL-Back")


        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_rho={}_s1.png".format(param,rholist_complex[hyperparam]),format='png')
        plt.clf()
        
        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex-config_ip.mreturn_o_complex,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2),marker="v",markersize=10,color = '#1f77b4', label = "ECBias^2-KL")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex-config_ip.mreturn_o_complex,axis=1),2),axis=0),marker="^",markersize=10,color='#1f77b4',label = "ECVar-KL")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_rho={}_s2.png".format(param,rholist_complex[hyperparam]),format='png')
        plt.clf()
        
        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb-config_ip.mreturn_o_complex_wb,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed',color='#ff7f0e',label = "ECMSE-WB")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2),marker="v",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECBias^2-WB")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb-config_ip.mreturn_o_complex_wb,axis=1),2),axis=0),marker="^",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECVar-WB")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_rho={}_s3.png".format(param,rholist_complex[hyperparam]),format='png')
        plt.clf()

        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general-config_ip.mreturn_o_complex_wb_general,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#2ca02c',label = "ECMSE-WB-Full")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2),marker="v",markersize=10,color='#2ca02c',linestyle='-.',label = "ECBias^2-WB-Full")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general-config_ip.mreturn_o_complex_wb_general,axis=1),2),axis=0),marker="^",markersize=10,color='#2ca02c',linestyle='-.',label = "ECVar-WB-Full")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_rho={}_s4.png".format(param,rholist_complex[hyperparam]),format='png')
        plt.clf()
        
        plt.plot(deltalist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back-config_ip.mreturn_o_complex_kl_back,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#B8860B',label = "ECMSE-KL-Back")
        plt.plot(deltalist_complex,np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2),marker="v",markersize=10,color='#B8860B',linestyle='-.',label = "ECBias^2-KL-Back")
        plt.plot(deltalist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back-config_ip.mreturn_o_complex_kl_back,axis=1),2),axis=0),marker="^",markersize=10,color='#B8860B',linestyle='-.',label = "ECVar-KL-Back")
        plt.xlabel(r'$\delta/\delta_{max}$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_rho={}_s5.png".format(param,rholist_complex[hyperparam]),format='png')
        plt.clf()

    if param == 'rho':
        
        for k in range(config_ip.n_experiment):
            print(k)
            data = config_ip.fulldata
            mask = utils_ip.missing()
            meanlist,covariancelist = utils_ip.individualposterior(data, mask, config_ip.mu_p, config_ip.covp_inv, config_ip.cov)
    
            deltalist_complex  = np.linspace(0.000, 1.0, num = config_ip.num_delta)
            rholist_complex = np.linspace(0.001,5.0, num = config_ip.num_rho)#rho should not equal to 0.
        
            for i in range(config_ip.num_rho):
            
                _,final_mu,final_sigma = utils_ip.consensusforwardkl(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],rholist_complex[i])
                completed_data = utils_ip.imputation(data,mask,final_mu,final_sigma)       

                _,final_mu_wb,final_sigma_wb = utils_ip.consensuswasserstein(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],rholist_complex[i],'l2-norm')
                completed_data_wb = utils_ip.imputation(data,mask,final_mu_wb,final_sigma_wb)
            
                _,final_mu_wb_general,final_sigma_wb_general = utils_ip.consensuswasserstein_general(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],rholist_complex[i],'l2-norm')
                completed_data_wb_general = utils_ip.imputation(data,mask,final_mu_wb_general,final_sigma_wb_general)
            
                _,final_mu_kl_back,final_sigma_kl_back = utils_ip.consensusbackwardkl(meanlist,covariancelist,deltalist_complex[hyperparam],meanlist[0],np.full(len(meanlist),1/len(meanlist)),10**(-15),50,rholist_complex[i],'l1-norm')
                completed_data_kl_back = utils_ip.imputation(data,mask,final_mu_kl_back,final_sigma_kl_back)
            
                for j in range(config_ip.m):
                    _, _,config_ip.mreturn_i_complex[k,j,i], config_ip.mreturn_o_complex[k,j,i]  = utils_ip.Greedy(completed_data[j])
                    _, _,config_ip.mreturn_i_complex_wb[k,j,i], config_ip.mreturn_o_complex_wb[k,j,i]  = utils_ip.Greedy(completed_data_wb[j])
                    _, _,config_ip.mreturn_i_complex_wb_general[k,j,i], config_ip.mreturn_o_complex_wb_general[k,j,i]  = utils_ip.Greedy(completed_data_wb_general[j])
                    _, _,config_ip.mreturn_i_complex_kl_back[k,j,i], config_ip.mreturn_o_complex_kl_back[k,j,i]  = utils_ip.Greedy(completed_data_kl_back[j])
                
        sds_r = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_rho))
        for i in range(config_ip.num_rho):
            sds_r[:,i] = config_ip.mreturn_i_complex[:,:,i].flatten()
    
        sds_o_r = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_rho))
        for i in range(config_ip.num_rho):
            sds_o_r[:,i] = config_ip.mreturn_o_complex[:,:,i].flatten()

        sds_r_wb = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_rho))
        for i in range(config_ip.num_rho):
            sds_r_wb[:,i] = config_ip.mreturn_i_complex_wb[:,:,i].flatten()

        sds_o_r_wb = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_rho))
        for i in range(config_ip.num_rho):
            sds_o_r_wb[:,i] = config_ip.mreturn_o_complex_wb[:,:,i].flatten()

        sds_r_wb_general = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_rho))
        for i in range(config_ip.num_rho):
            sds_r_wb_general[:,i] = config_ip.mreturn_i_complex_wb_general[:,:,i].flatten()

        sds_o_r_wb_general = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_rho))
        for i in range(config_ip.num_rho):
            sds_o_r_wb_general[:,i] = config_ip.mreturn_o_complex_wb_general[:,:,i].flatten()
    
        sds_r_kl_back = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_rho))
        for i in range(config_ip.num_rho):
            sds_r_kl_back[:,i] = config_ip.mreturn_i_complex_kl_back[:,:,i].flatten()

        sds_o_r_kl_back = np.zeros((config_ip.n_experiment * config_ip.m ,  config_ip.num_rho))
        for i in range(config_ip.num_rho):
            sds_o_r_kl_back[:,i] = config_ip.mreturn_o_complex_kl_back[:,:,i].flatten()
            
        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex-config_ip.mreturn_o_complex,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")

        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb-config_ip.mreturn_o_complex_wb,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed', label = "ECMSE-WB")

        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general-config_ip.mreturn_o_complex_wb_general,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-WB-Full")

        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back-config_ip.mreturn_o_complex_kl_back,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.', label = "ECMSE-KL-Back")


        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_delta={}_s1.png".format(param,deltalist_complex[hyperparam]),format='png')
        plt.clf()
        
        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex-config_ip.mreturn_o_complex,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r - sds_o_r,axis=0),0),2)),marker="o",markersize=10,label = "ECMSE-KL")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2),marker="v",markersize=10,color = '#1f77b4', label = "ECBias^2-KL")
        plt.plot(rholist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex-config_ip.mreturn_o_complex,axis=1),2),axis=0),marker="^",markersize=10,color='#1f77b4',label = "ECVar-KL")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_delta={}_s2.png".format(param,deltalist_complex[hyperparam]),format='png')
        plt.clf()
        
        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb-config_ip.mreturn_o_complex_wb,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2)),marker="p",markersize=10,linestyle='dashed',color='#ff7f0e',label = "ECMSE-WB")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_wb - sds_o_r_wb,axis=0),0),2),marker="v",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECBias^2-WB")
        plt.plot(rholist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb-config_ip.mreturn_o_complex_wb,axis=1),2),axis=0),marker="^",markersize=10,color='#ff7f0e',linestyle='dashed',label = "ECVar-WB")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_delta={}_s3.png".format(param,deltalist_complex[hyperparam]),format='png')
        plt.clf()

        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general-config_ip.mreturn_o_complex_wb_general,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#2ca02c',label = "ECMSE-WB-Full")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_wb_general - sds_o_r_wb_general,axis=0),0),2),marker="v",markersize=10,color='#2ca02c',linestyle='-.',label = "ECBias^2-WB-Full")
        plt.plot(rholist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_wb_general-config_ip.mreturn_o_complex_wb_general,axis=1),2),axis=0),marker="^",markersize=10,color='#2ca02c',linestyle='-.',label = "ECVar-WB-Full")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_delta={}_s4.png".format(param,deltalist_complex[hyperparam]),format='png')
        plt.clf()
        
        plt.plot(rholist_complex,(np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back-config_ip.mreturn_o_complex_kl_back,axis=1),2),axis=0)+np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2)),marker="s",markersize=10,linestyle='-.',color='#B8860B',label = "ECMSE-KL-Back")
        plt.plot(rholist_complex,np.power(np.maximum(np.mean(sds_r_kl_back - sds_o_r_kl_back,axis=0),0),2),marker="v",markersize=10,color='#B8860B',linestyle='-.',label = "ECBias^2-KL-Back")
        plt.plot(rholist_complex,np.mean(np.power(np.std(config_ip.mreturn_i_complex_kl_back-config_ip.mreturn_o_complex_kl_back,axis=1),2),axis=0),marker="^",markersize=10,color='#B8860B',linestyle='-.',label = "ECVar-KL-Back")
        plt.xlabel(r'$\rho$',fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(config_ip.epspath + "/{}_delta={}_s5.png".format(param,deltalist_complex[hyperparam]),format='png')
        plt.clf()
