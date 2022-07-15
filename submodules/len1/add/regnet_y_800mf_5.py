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

    def forward(self, x89, x103):
        x104=operator.add(x89, x103)
        return x104

m = M().eval()
x89 = torch.randn(torch.Size([1, 320, 14, 14]))
x103 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x89, x103)
end = time.time()
print(end-start)
