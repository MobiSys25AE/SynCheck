import torch

class GaussianTransform(object):
    def __init__(self, weak_mean=0.0, weak_std=0.01, strong_mean=0.0, strong_std=0.05, seed=420):
        self.seed = seed
        self.weak_mean = weak_mean
        self.weak_std = weak_std
        self.strong_mean = strong_mean
        self.strong_std = strong_std
        torch.manual_seed(self.seed)
    
    def __call__(self, x):
        # add noise to weakly augmented signal
        weak_noise = torch.normal(mean=self.weak_mean, std=self.weak_std, size=x.size())
        x_weak = x + weak_noise
        # add noise to strongly augmented signal
        strong_noise = torch.normal(mean=self.strong_mean, std=self.strong_std, size=x.size())
        x_strong = x + strong_noise
        return x_weak, x_strong, x
        
        