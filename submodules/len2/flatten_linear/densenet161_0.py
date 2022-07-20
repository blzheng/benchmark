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
        self.linear0 = Linear(in_features=2208, out_features=1000, bias=True)

    def forward(self, x569):
        x570=torch.flatten(x569, 1)
        x571=self.linear0(x570)
        return x571

m = M().eval()
x569 = torch.randn(torch.Size([1, 2208, 1, 1]))
start = time.time()
output = m(x569)
end = time.time()
print(end-start)
