import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskNeedsMLP(nn.Module):
    def __init__(self, in_dim, trunk=(64, 32), head=(16,), p=0.25):
        super().__init__()
        layers, d = [], in_dim
        
        # Shared representations
        for h in trunk:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(p)]
            d = h
        self.trunk = nn.Sequential(*layers)
        
        # Task-specific heads
        def make_head():
            L, d_h = [], d
            for h in head:
                L += [nn.Linear(d_h, h), nn.GELU(), nn.Dropout(p)]
                d_h = h
            L += [nn.Linear(d_h, 1)] # Outputs logits, not probabilities
            return nn.Sequential(*L)
            
        self.head_accum = make_head()
        self.head_income = make_head()

    def forward(self, x):
        z = self.trunk(x)
        return self.head_accum(z).squeeze(-1), self.head_income(z).squeeze(-1)

def multi_task_loss(logits_a, logits_i, y_a, y_i, w_a, w_i, joint_prior, lam=0.1):
    """
    Combines weighted Binary Cross Entropy with a KL-Divergence penalty 
    to enforce alignment with macro-business segments.
    """
    bce_a = F.binary_cross_entropy_with_logits(logits_a, y_a, pos_weight=w_a)
    bce_i = F.binary_cross_entropy_with_logits(logits_i, y_i, pos_weight=w_i)
    
    p_a = torch.sigmoid(logits_a)
    p_i = torch.sigmoid(logits_i)
    
    # Calculate implied joint distribution of the batch
    implied = torch.stack([
        ((1 - p_a) * (1 - p_i)).mean(), # Neither
        ((1 - p_a) * p_i).mean(),       # Income Only
        (p_a * (1 - p_i)).mean(),       # Accumulation Only
        (p_a * p_i).mean()              # Both
    ])
    
    # Penalize deviation from actual business distribution
    consistency = F.kl_div((implied + 1e-8).log(), joint_prior, reduction='batchmean')
    
    return bce_a + bce_i + lam * consistency