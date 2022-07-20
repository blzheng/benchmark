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
        self.linear0 = Linear(in_features=912, out_features=1000, bias=True)

    def forward(self, x192):
        x193=x192.flatten(start_dim=1)
        x194=self.linear0(x193)
        return x194

m = M().eval()
x192 = torch.randn(torch.Size([1, 912, 1, 1]))
start = time.time()
output = m(x192)
end = time.time()
print(end-start)
