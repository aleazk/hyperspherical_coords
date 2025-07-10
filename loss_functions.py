# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:52:24 2024

@author: asc007
"""

import torch
import SC_vectorized as SC

def Phi_loss_function_mu(mu, device, compress):
    
    d = mu.size(1)
    
    # cos_phi_var part ----------
    
    cos_phi_var = torch.sqrt(SC.cart_to_cos_sph (mu, device).var(dim=0) + 0.0001)
    
    cos_phi_var_p = torch.squeeze(SC.cart_to_cos_sph (torch.unsqueeze(torch.ones(d).to(device).float(), dim=0), device)
                                  , dim=0)#*0  # <--------- Prior for cos_phi_var
    
    R = torch.sqrt(torch.tensor([d]).to(device))
    
    Beta = 50 * R/(torch.arange(1, d).to(device)).pow(1/2)
    
    KLD_cos_phi_var = (((cos_phi_var+1-cos_phi_var_p).pow(2) - (cos_phi_var+1-cos_phi_var_p).pow(2).log() - 1)
                   * Beta).sum()
    
    # cos_phi_mean part ----------

    cos_phi_mean = SC.cart_to_cos_sph (mu, device).mean(dim=0)
    
    if compress=='half':
        cos_phi_mean_p = torch.squeeze(SC.cart_to_cos_sph (torch.unsqueeze(torch.ones(d).to(device).float(), dim=0), device)
                                       , dim=0)#*0 #+ 1.0       # <--------- Prior for cos_phi_mean
    elif compress=='full':
        cos_phi_mean_p = torch.squeeze(SC.cart_to_cos_sph (torch.unsqueeze(torch.ones(d).to(device).float(), dim=0), device)
                                       , dim=0)*0 + 1.0        # <--------- Prior for cos_phi_mean

    KLD_cos_phi_mean = ((cos_phi_mean_p - cos_phi_mean).pow(2) * Beta).sum()
    
    # cos_phi total ----------

    KLD_cos_phi = 1 * KLD_cos_phi_var + 0.05 * KLD_cos_phi_mean * 1 
    #                                                           # * 1
    return KLD_cos_phi


def r_loss_function_mu(mu, device):
    
    d = mu.size(1)
    
    # r_var part ----------
    
    r_var = torch.sqrt((SC.r(mu)).var(dim=0) + 0.0001)
    
    r_var_p = torch.sqrt(torch.tensor([0.5]).to(device))  # <--------- Prior for r_var
    
    #KLD_r_var = (r_var/r_var_p).pow(2) - (r_var/r_var_p).pow(2).log() - 1
    
    KLD_r_var = (r_var+1).pow(2) - (r_var+1).pow(2).log() - 1   # For r_var_p=0
    
    # r_mean part ----------
    
    r_mean = SC.r(mu).mean()
    
    r_mean_p = torch.sqrt(torch.tensor([d-0.5]).to(device))#*0   # <--------- Prior for r_mean
  
    KLD_r_mean = (r_mean - r_mean_p).pow(2)
    
    # r total ----------
    
    KLD_r = 50 * KLD_r_var + 50 * KLD_r_mean
    
    return KLD_r


def Phi_loss_function_sigma(logvar, device):
    
    d = logvar.size(1)
    
    std = logvar.mul(0.5).exp_()
    
    # cos_phi_sigma_var part ----------
    
    cos_phi_sigma_var = torch.sqrt(SC.cart_to_cos_sph (std, device).var(dim=0) + 0.0001)
    
    cos_phi_sigma_var_p = torch.squeeze(SC.cart_to_cos_sph (torch.unsqueeze(torch.ones(d).to(device).float(), dim=0), device)
                                  , dim=0)*0  # <--------- Prior for cos_phi_sigma_var
    
    R = torch.sqrt(torch.tensor([d]).to(device))
    
    Beta = 50 * R/(torch.arange(1, d).to(device)).pow(1/2)
    
    KLD_cos_phi_sigma_var = (((cos_phi_sigma_var+1-cos_phi_sigma_var_p).pow(2) - 
                              (cos_phi_sigma_var+1-cos_phi_sigma_var_p).pow(2).log() - 1) * Beta).sum()
    
    # cos_phi_sigma_mean part ----------
    
    cos_phi_sigma_mean = SC.cart_to_cos_sph (std, device).mean(dim=0)
    
    cos_phi_sigma_mean_p = torch.squeeze(SC.cart_to_cos_sph (torch.unsqueeze(torch.ones(d).to(device).float(), dim=0), device)
                                  , dim=0)#*0 #+1          # <--------- Prior for cos_phi_sigma_mean

    KLD_cos_phi_sigma_mean = ((cos_phi_sigma_mean_p - cos_phi_sigma_mean).pow(2) * Beta).sum()
    
    # cos_phi_sigma total ----------

    KLD_cos_phi_sigma = 1 * KLD_cos_phi_sigma_var + 0.05 * KLD_cos_phi_sigma_mean
    
    return KLD_cos_phi_sigma


def r_loss_function_sigma(logvar, device):
    
    d = logvar.size(1)
    
    std = logvar.mul(0.5).exp_()
    
    # r_sigma_var part ----------
    
    r_sigma_var = torch.sqrt((SC.r(std)).var(dim=0) + 0.0001)
    
    r_sigma_var_p = torch.sqrt(torch.tensor([0.5]).to(device))  # <--------- Prior for r_sigma_var
    
    KLD_r_sigma_var = (r_sigma_var/r_sigma_var_p).pow(2) - (r_sigma_var/r_sigma_var_p).pow(2).log() - 1
    
    #KLD_r_sigma_var = (r_sigma_var+1).pow(2) - (r_sigma_var+1).pow(2).log() - 1   # For r_sigma_var_p=0
    
    # r_sigma_mean part ----------
    
    r_sigma_mean = SC.r(std).mean()
    
    r_sigma_mean_p = torch.sqrt(torch.tensor([d-0.5]).to(device))*1.0 # <--------- Prior for r_sigma_mean
  
    KLD_r_sigma_mean = (r_sigma_mean - r_sigma_mean_p).pow(2)
    
    # r_sigma total ----------
    
    KLD_r_sigma = 50 * KLD_r_sigma_var + 50 * KLD_r_sigma_mean
    
    return KLD_r_sigma
