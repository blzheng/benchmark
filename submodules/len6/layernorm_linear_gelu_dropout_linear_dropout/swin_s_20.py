import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.layernorm44 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear42 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu20 = GELU(approximate='none')
        self.dropout40 = Dropout(p=0.0, inplace=False)
        self.linear43 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout41 = Dropout(p=0.0, inplace=False)

    def forward(self, x494):
        x495=self.layernorm44(x494)
        x496=self.linear42(x495)
        x497=self.gelu20(x496)
        x498=self.dropout40(x497)
        x499=self.linear43(x498)
        x500=self.dropout41(x499)
        return x500

m = M().eval()
x494 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x494)
end = time.time()
print(end-start)
