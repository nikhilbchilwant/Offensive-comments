from torch import nn
import torch
from torch_two_sample import MMDStatistic

class domain_shift_loss(nn.Module):
    def __init__(self, source_size, target_size, bandwidth=2):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mmd_loss = MMDStatistic(source_size, target_size)
        #https://discuss.pytorch.org/t/how-to-learn-the-weights-between-two-losses/39681/2?u=nikhilbchilwant
        self.lam = nn.Parameter(torch.ones(1))
        self.bandwidth = bandwidth
        self.source_size = source_size
        self.target_size = target_size


    def forward(self, output, target, latent_source, latent_target):
        result = self.cross_entropy_loss(output, target)
        result = result * self.lam

        #For the last iteration of an epoch, the batch may contain
        #less no. of samples than self.source_size
        #The below solution didn't work as it gives warning:
        #The .grad attribute of a Tensor that is not a leaf Tensor is being accessed
        # if latent_source.shape[0] != self.source_size:
        #     self.mmd_loss = MMDStatistic(latent_source.shape[0], self.target_size)
        #     self.source_size = latent_source.shape[0]
        #
        #No solution implemented yet. As a result, this will not work properly when
        #no. of samples skipped is going to be comparable with the training dataset.

        #The function only supports 2D tensors. So, *hopefully* I'm changing the dimensions in the right way.
        if latent_source.shape[0] == self.source_size:
            result = result + self.mmd_loss(latent_source.view(self.source_size, 128*768), 
                                latent_target.view(self.target_size, 128*768), alphas=[1. / self.bandwidth],
                                    ret_matrix=False)
                                    #ret_matrix returns kernels

        return result