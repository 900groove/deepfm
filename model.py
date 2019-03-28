import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler


class DeepFM(nn.Module):
    def __init__(self,
                 feature_size,
                 embedding_size=4,
                 hidden_dim=[64, 32],
                 num_classes=[2],
                 dropout=[0.5, 0.5],):
        
        super().__init__()
        self.field_size = len(feature_size)
        self.feature_size = feature_size
        self.emb_size = embedding_size
        self.hidden_dims = hidden_dim
        self.num_classes = num_classes
        self.bias = torch.nn.Parameter(torch.randn(1))
        
        # factorization machine
        self.fm_first_order_emb = nn.ModuleList([nn.Embedding(size, 1) for size in self.feature_size])
        self.fm_second_order_emb = nn.ModuleList([nn.Embedding(size, self.emb_size) for size in self.feature_size])
        
        # neural network
        all_dims = [self.field_size * self.emb_size] + self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dim) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i-1], all_dims[i]))
            setattr(self, 'relu_' + str(i), F.relu)
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i-1]))
            
    def forward(self, Xi, Xv):
        # factorization machine
        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t()) for i, emb in enumerate(self.fm_first_order_emb)]        
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_emb)]
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb
        fm_second_order_emb_square = [item*item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        
        # neural network
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'relu_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
            
        total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + torch.sum(deep_out, 1) + self.bias
        
        return total_sum