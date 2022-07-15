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

    def forward(self, x247, x233):
        x248=operator.add(x247, x233)
        return x248

m = M().eval()
x247 = torch.randn(torch.Size([1, 144, 14, 14]))
x233 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x247, x233)
end = time.time()
print(end-start)
