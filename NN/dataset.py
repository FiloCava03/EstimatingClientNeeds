import torch
from torch.utils.data import Dataset

class FinancialNeedsDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.05, continuous_mask=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y_inc = torch.tensor(y['IncomeInvestment'].values, dtype=torch.float32)
        self.y_acc = torch.tensor(y['AccumulationInvestment'].values, dtype=torch.float32)
        
        self.augment = augment
        self.noise_std = noise_std
        # A boolean mask telling the dataset which columns are safe to perturb
        self.continuous_mask = torch.tensor(continuous_mask, dtype=torch.bool) if continuous_mask is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_val = self.X[idx].clone() # Clone to avoid modifying the underlying dataset
        
        # Targeted Tabular Augmentation: Only apply noise to continuous features
        if self.augment and self.continuous_mask is not None:
            noise = torch.randn(self.continuous_mask.sum()) * self.noise_std
            x_val[self.continuous_mask] += noise
            
        return x_val, self.y_inc[idx], self.y_acc[idx]