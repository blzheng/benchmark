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
        self.linear0 = Linear(in_features=1920, out_features=1000, bias=True)

    def forward(self, x709):
        x710=torch.flatten(x709, 1)
        x711=self.linear0(x710)
        return x711

m = M().eval()
x709 = torch.randn(torch.Size([1, 1920, 1, 1]))
start = time.time()
output = m(x709)
end = time.time()
print(end-start)
