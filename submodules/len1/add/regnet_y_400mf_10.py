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

    def forward(self, x171, x185):
        x186=operator.add(x171, x185)
        return x186

m = M().eval()
x171 = torch.randn(torch.Size([1, 440, 7, 7]))
x185 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x171, x185)
end = time.time()
print(end-start)
