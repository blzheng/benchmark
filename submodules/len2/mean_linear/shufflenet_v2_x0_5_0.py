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
        self.linear0 = Linear(in_features=1024, out_features=1000, bias=True)

    def forward(self, x365):
        x366=x365.mean([2, 3])
        x367=self.linear0(x366)
        return x367

m = M().eval()
x365 = torch.randn(torch.Size([1, 1024, 7, 7]))
start = time.time()
output = m(x365)
end = time.time()
print(end-start)
