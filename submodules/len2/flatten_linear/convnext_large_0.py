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
        self.flatten0 = Flatten(start_dim=1, end_dim=-1)
        self.linear72 = Linear(in_features=1536, out_features=1000, bias=True)

    def forward(self, x426):
        x427=self.flatten0(x426)
        x428=self.linear72(x427)
        return x428

m = M().eval()
x426 = torch.randn(torch.Size([1, 1536, 1, 1]))
start = time.time()
output = m(x426)
end = time.time()
print(end-start)
