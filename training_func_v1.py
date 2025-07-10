"""
Created on Tue Apr 26 09:09:33 2022

Edited on Wed Dec 13 00:00:00 2023

v1: - implement VAE 

@author: sal117

Edits:

@author: asc007
"""

# imports
# torch and friends
import torch
import torch.nn.functional as F

# standard
import numpy as np
from tqdm import tqdm

# custom
import SC_vectorized as SC
import loss_functions as LF


# === sandbox version
def calc_kl(logvar, mu, mu_p=0.0, logvar_p=0, reduce='mean',option=None, epoch=1, dataset='mnist', variant_exp=0, compress='zero'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param reduce: type of reduce: 'sum', 'none'
    :param option: 0=standard KLD, 1=KLDAle
    :return: kld
    """

    # --- take into account where the center of the latent should be
    if not isinstance(mu_p, torch.Tensor):
        mu_p = torch.tensor(mu_p).to(mu.device)
        error_mu    = mu - mu_p
    else:
        error_mu    = mu
    if not isinstance(logvar_p, torch.Tensor):
        logvar_p    = torch.tensor(logvar_p).to(mu.device)
        
    if option == 'KLDAle':
        
        KLD_phi_mu = LF.Phi_loss_function_mu(mu, mu.device, compress)
        
        KLD_r_mu = LF.r_loss_function_mu(mu, mu.device)
        
        KLD_phi_sigma = LF.Phi_loss_function_sigma(logvar, mu.device)
        
        KLD_r_sigma = LF.r_loss_function_sigma(logvar, mu.device)
        if dataset=='mnist':
            
            if variant_exp==0:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 50000 * KLD_phi_sigma + 500 * KLD_r_sigma
            elif variant_exp==2:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 1*50000 * KLD_phi_sigma + 500 * KLD_r_sigma
            elif variant_exp==1:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 1*50000 * KLD_phi_sigma + 500 * KLD_r_sigma
        elif dataset=='fmnist':
            
            if variant_exp==0:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 4.5*50000 * KLD_phi_sigma + 500 * KLD_r_sigma
            elif variant_exp==2:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 4.5*50000 * KLD_phi_sigma + 500 * KLD_r_sigma
            elif variant_exp==1:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 4.5*50000 * KLD_phi_sigma + 500 * KLD_r_sigma         
        elif dataset=='cifar10':
            
            if variant_exp==0:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma    
            elif variant_exp==2:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 1*500000 * KLD_phi_sigma + 500 * KLD_r_sigma
            elif variant_exp==1:
                kl = 1 * 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma    
        elif dataset=='svhn':
            
            if variant_exp==0:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma
            elif variant_exp==2:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 1*500000 * KLD_phi_sigma + 500 * KLD_r_sigma
            elif variant_exp==1:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 1*500000 * KLD_phi_sigma + 500 * KLD_r_sigma
        elif dataset=='celeba64':
            
            kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma
        elif dataset=='celeba64_sans_glasses':
            
            if variant_exp==0:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma    
            elif variant_exp==2:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 1*500000 * KLD_phi_sigma + 500 * KLD_r_sigma
            elif variant_exp==1:
                kl = 1 * 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma 
                #    1 * 
        elif dataset=='galaxy_zoo128':
            
            if variant_exp==0:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma    
            elif variant_exp==2:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 1*500000 * KLD_phi_sigma + 500 * KLD_r_sigma
            elif variant_exp==1:
                kl = 1 * 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma 
        elif dataset=='galaxy_zoo256':
            
            if variant_exp==0:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma    
            elif variant_exp==2:
                kl = 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 1*500000 * KLD_phi_sigma + 500 * KLD_r_sigma
            elif variant_exp==1:
                kl = 1 * 1000 * KLD_phi_mu + 50 * 6 * KLD_r_mu + 500000 * KLD_phi_sigma + 500 * KLD_r_sigma 
    else:
        # --- compute standard KLD with prior using mean over z_dim
        kl = -0.5 * (1 + logvar - logvar_p - logvar.exp() / torch.exp(logvar_p) - error_mu.pow(2) / torch.exp(logvar_p)).mean(1)

    if reduce == 'sum': # sum over the batch
        kl = torch.sum(kl)
    elif reduce == 'mean': # mean over the batch
        kl = torch.mean(kl)
    return kl


def reparameterize(mu, logvar, Normalize_z=False): 
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device  = mu.device
    std     = torch.exp(0.5 * logvar)
    eps     = torch.randn_like(std).to(device)
    R = torch.sqrt(torch.tensor([mu.size(1)]).to(device))
    if Normalize_z:
        u       = mu + eps * std
        z = R * u/torch.unsqueeze(SC.r(u), dim=1)
    else:
        z       = mu + eps * std
    return z


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """
    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1) # over the pixels
        if reduction == 'sum': # over the batch
            recon_error = recon_error.sum()
        elif reduction == 'mean': # over the batch
            recon_error = recon_error.mean() # unit should be std in % of range ([0,1])
        else:
            raise NotImplementedError
    elif loss_type == 'l1':
        # gain 100 is for achieving same order of magnitude with MSE
        recon_error = 100*F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        # gain 10 is for achieving same order of magnitude with MSE
        recon_error = 10*F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


# ===================================== train_VAE


def train_VAE(H,*args):

    if len(args) == 0:
        return ['DT (RHS)', 'EDT', 'Ekl', 'DDT']

    elif len(args) == 7:
        model               = args[0]
        train_data_loader   = args[1]
        optimizer_e         = args[2]
        e_scheduler         = args[3]
        optimizer_d         = args[4]
        d_scheduler         = args[5]
        epoch               = args[6]

    else:
        raise Exception("!!!!!!!!! not enough arguments")

    model.train()

    perf            = np.zeros((1,len(H.perf_dict)-2)) # where to store the tracking of the performance and losses
    batch_loss1     = []
    batch_loss2     = []

    print(' --- Training epoch: ' + str(epoch+1))
    pbar = tqdm(iterable=train_data_loader, unit="batch")
    idx = 0

    for batch in pbar:
        idx += 1
        if idx>H.max_batch: # to save time for debugging
            break

        # --------------train------------
        if H.dataset in ["cifar10", "svhn", "fmnist", "mnist", "celeba32", "celeba64", "celeba128", 'celeba64_sans_glasses', 'galaxy_zoo128', 'galaxy_zoo256']:
            
            cond  = batch[1] 
               
            batch = batch[0]
            
        else:
            cond=None

        #  =============================  VAEKTORY training start here

        if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)
                
        if H.variant_exp == 1:
            X = batch.to(H.device)
            X = X[cond==H.c_N, :, :, :]
            
        elif H.variant_exp == 2:
            X = batch.to(H.device)
            X = X[cond!=H.c_A, :, :, :]
            
        else:
            X = batch.to(H.device)

            # =========== Update E ================
           
        for param in model.encoder.main.parameters():
                     param.requires_grad = True
        for param in model.decoder.main.parameters():
                     param.requires_grad = False
        for param in model.encoder.F2L.parameters():
                     param.requires_grad = True
        for param in model.decoder.L2F.parameters():
                     param.requires_grad = False

        # --- process x, standard VAE
        mu, logvar, _ = model.encode(X)
        # --- process Xz
        if H.no_rep_trick:
            z = mu
        else:
            z           = reparameterize(mu, logvar, Normalize_z=H.Normalize_z)
        Xz, _       = model.decode(z)
        # --- compute Data loss and KL loss (typical VAE)
        EDT     = calc_reconstruction_loss(X, Xz, loss_type=H.recon_loss_type, reduction=H.kld_red_E_VAE)

        Ekl     = calc_kl(logvar, mu, reduce=H.kld_red_E_VAE, logvar_p=H.mu_p_logvar , option=H.kld_option_mu,
                          epoch=epoch, dataset=H.dataset, variant_exp=H.variant_exp, compress=H.compress)
        # --- complete cost   
        Eloss           =  0.001*(H.beta_EDT * EDT +
                                  H.beta_Ekl * Ekl*(epoch**0.5 + 1)
                                  )
        if H.compress=='full':
            if epoch>100:
                Eloss           =  0.001*(H.beta_EDT * EDT +
                                          H.beta_Ekl * Ekl
                                          )
            elif epoch>300:
                Eloss           =  0.001*(H.beta_EDT * EDT +
                                          H.beta_Ekl * Ekl*((epoch-301)**0.5 + 1)
                                          )
            elif epoch>500:
                Eloss           =  0.001*(H.beta_EDT * EDT +
                                          H.beta_Ekl * Ekl
                                          )
            elif epoch>700:
                Eloss           =  0.001*(H.beta_EDT * EDT +
                                          H.beta_Ekl * Ekl*((epoch-701)**0.5 + 1)
                                          )
            elif epoch>900:
                Eloss           =  0.001*(H.beta_EDT * EDT +
                                          H.beta_Ekl * Ekl
                                          )
        elif H.kld_option_mu == 'KLD':
            Eloss           =  0.001*(H.beta_EDT * EDT +
                                      H.beta_Ekl * Ekl
                                      )
        # --- backpropagation of encoder
        optimizer_e.zero_grad()
        Eloss.backward()
        optimizer_e.step()

        # =========== Update D ================
        for param in model.encoder.main.parameters():
                     param.requires_grad = False
        for param in model.decoder.main.parameters():
                     param.requires_grad = True
        for param in model.encoder.F2L.parameters():
                     param.requires_grad = False
        for param in model.decoder.L2F.parameters():
                     param.requires_grad = True

        Xz, _   = model.decode(z.detach())
        # --- standard data term
        DDT     = calc_reconstruction_loss(X, Xz, loss_type=H.recon_loss_type, reduction="mean")
        Dloss           =  0.001*(H.beta_DDT * DDT)
        # --- backpropagation of Decoder
        optimizer_d.zero_grad()
        Dloss.backward()
        optimizer_d.step()

        if torch.isnan(Dloss) or torch.isnan(Eloss):
            print(f'Eloss: {Eloss:.5f}')
            print(f'EDT: {EDT:.3f}, Ekl: {Ekl:.3f} ')
            print(f'Dloss: {Dloss:.5f}')
            print(f'DDT: {DDT:.3f}')
            raise ValueError('!!! nan detected in Dloss or Eloss during training')
        # pbar.set_description_str('epoch #{}'.format(epoch))
        pbar.set_postfix(Eloss=Eloss.data.cpu().item(), Dloss=Dloss.data.cpu().item())
        # ======== update the losses for display
        batch_loss1.append(EDT.data.cpu().item())
        batch_loss2.append(Ekl.data.cpu().item())

    e_scheduler.step()
    d_scheduler.step()

    pbar.close()

    perf[0,0] = 100*((H.scale*np.mean(batch_loss1))**0.5)  # normalise as % of dynamic range (0,1)
    perf[0,3] = np.mean(batch_loss1)

    # --- Autotune
    if epoch==0:
        l = 0.0
    else:
        l = 1.0 # if l<1 then it is adaptative autotune (adapt beta at each iteration)
    loss1               = np.mean(batch_loss1)
    H.beta_Ekl          = l*H.beta_Ekl + (1-l) * H.betat_Ekl * loss1 / (np.mean(batch_loss2)+1e-3)

    # --- epoch summary
    print( ' --- NRSB-VAE '+ H.version + ' Summary:')
    print(f'beta_Ekl: {H.beta_Ekl:.3f}')
    print(f'Eloss: {Eloss:.5f}')
    print(f'EDT: {EDT:.3f}, Ekl: {Ekl:.3f} ')
    print(f'Dloss: {Dloss:.5f}')
    print(f'DDT: {DDT:.3f}')

    return perf