import torch


def r (x):

    r = torch.linalg.norm (x, dim=1)

    return r


def cart_to_cos_sph (x, device):
    
    m = x.size(0)
    
    n = x.size(1)
    
    mask = torch.triu(torch.ones(n, n)).to(device)
    
    mask = torch.unsqueeze(mask, dim=0)
    
    mask = mask.expand(m, n, n)
    
    X = torch.unsqueeze(x, dim=1).expand(m, n, n)
    
    X_squared = torch.square(X)
    
    X_squared_masked = X_squared * mask
    
    denom = torch.sqrt(torch.sum(X_squared_masked, dim=2)+0.001)
    
    cos_phi = x / denom
    
    return cos_phi[:, 0:n-1]


def cart_to_sin_sph (x, device):
    
    return torch.sqrt (1 - cart_to_cos_sph (x, device).pow(2))


def cart_to_sph (x, device):
    
    m = x.size(0)
    
    n = x.size(1)
    
    mask = torch.triu(torch.ones(n, n)).to(device)
    
    mask = torch.unsqueeze(mask, dim=0)
    
    mask = mask.expand(m, n, n)
    
    X = torch.unsqueeze(x, dim=1).expand(m, n, n)
    
    X_squared = torch.square(X)
    
    X_squared_masked = X_squared * mask
    
    denom = torch.sqrt(torch.sum(X_squared_masked, dim=2)+0.001)
    
    phi_plus = torch.arccos (x / denom)
    
    phi_minus = 2*3.141592654 - phi_plus
    
    phi = phi_plus 
    
    phi[:, n-2] = torch.where (x[:, n-1] >= 0, phi_plus[:, n-2], phi_minus[:, n-2])
    
    return phi[:, 0:n-1]


def sph_to_cart (R, phi, device):

    m = phi.size(0)
    
    n = phi.size(1)+1
    
    mask = torch.tril(torch.ones(n-1, n-1)).to(device)
    
    mask = torch.unsqueeze(mask, dim=0)
    
    mask = mask.expand(m, n-1, n-1)
    
    PHI = torch.unsqueeze(phi, dim=1).expand(m, n-1, n-1)
    
    sin_PHI = torch.sin(PHI)
    
    mask_ = torch.unsqueeze(torch.triu(torch.ones(n-1, n-1), diagonal=1).to(device), dim=0).expand(m, n-1, n-1)
    
    sin_PHI_masked = sin_PHI * mask + mask_
    
    sin_prod = torch.prod (sin_PHI_masked, dim=2) 
    
    ones = torch.ones(m).to(device)
    
    sin_PROD = torch.column_stack((ones, sin_prod))
    
    cos_R = torch.mul (torch.column_stack((torch.cos (phi), ones)), torch.unsqueeze(R, dim=1))
    
    x = torch.mul (sin_PROD, cos_R) 
    
    return x


#X = np.array ([0, 1, 0])

#Y = torch.tensor ([[0, 0, 1], [0, -1, 0], [0, 0, -1], [0, 0.999, -0.001]])

#print (sph_to_cart (r(Y), cart_to_sph (Y,'cpu'), 'cpu'))
