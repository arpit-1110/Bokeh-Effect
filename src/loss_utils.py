import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def real_mse_loss(D_out):
    loss = nn.MSELoss()
    target_tensor = 1.0
    return loss(D_out, torch.tensor(target_tensor).expand_as(D_out).to(device))
    # return torch.mean((D_out-1)**2)

def fake_mse_loss(D_out):
    loss = nn.MSELoss()
    target_tensor = 0.0
    return loss(D_out, torch.tensor(target_tensor).expand_as(D_out).to(device))
    # return torch.mean(D_out**2)

def cycle_consistency_loss  (real_im, reconstructed_im, lambda_weight):
    reconstructed_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    return lambda_weight*reconstructed_loss
