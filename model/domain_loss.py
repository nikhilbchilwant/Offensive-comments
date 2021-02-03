from torch import nn
import torch
from torch_two_sample import MMDStatistic

class domain_shift_loss(nn.Module):
    def __init__(self, source_size, target_size, bandwidth=2):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mmd_loss = MMDStatistic(source_size, target_size)
        self.lam = nn.Parameter(torch.ones(1))
        self.bandwidth = bandwidth
        self.source_size = source_size
        self.target_size = target_size


    def forward(self, output, target, latent_source, latent_target):
        result = self.cross_entropy_loss(output, target)
        result = result * self.lam
        if latent_source.shape[0] == self.source_size:
            #The function only supports 2D tensors. So, *hopefully* I'm changing the dimensions in the right way.
            result = result + self.mmd_loss(latent_source.view(self.source_size, 128*768), 
                                latent_target.view(self.target_size, 128*768), alphas=[1. / self.bandwidth],
                                 ret_matrix=False)
                                 #ret_matrix returns kernels
        return result