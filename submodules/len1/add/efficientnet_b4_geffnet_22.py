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

    def forward(self, x427, x413):
        x428=operator.add(x427, x413)
        return x428

m = M().eval()
x427 = torch.randn(torch.Size([1, 272, 7, 7]))
x413 = torch.randn(torch.Size([1, 272, 7, 7]))
start = time.time()
output = m(x427, x413)
end = time.time()
print(end-start)
