import torch
from torch import nn, Tensor
import torch.nn.functional as F
from onebitllms import activation_quant_triton, weight_quant_triton

class BitLinear(nn.Module):
    """
    BitNet 1.58bit 방식 양자화 선형 계층
    가중치는 1.58bit (-1, 0, 1) 3값 양자화
    """
    def __init__(self, in_features: int, out_features: int, bias: bool=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: Tensor):
        w = self.weight
        print(f"[BitLinear] w.requires_grad: {w.requires_grad}, x.requires_grad: {x.requires_grad}")

        with torch.cuda.device(w.device):
            x_quant = x + (activation_quant_triton(x) - x).detach()
            w_quant = w + (weight_quant_triton(w) - w).detach()
            print(f"[BitLinear] w_quant unique: {w_quant.unique(sorted=True)}") # [-1, 0, 1] 인지 확인

        y = F.linear(x_quant, w_quant, bias=self.bias)
        
        return y
    
    def __repr__(self):
        return 'BitLinear(in_features={0}, out_features={1})'.format(self.in_features, self.out_features)