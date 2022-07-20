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

    def forward(self, x495, x491):
        x496=x495.sigmoid()
        x497=operator.mul(x491, x496)
        return x497

m = M().eval()
x495 = torch.randn(torch.Size([1, 1344, 1, 1]))
x491 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x495, x491)
end = time.time()
print(end-start)
