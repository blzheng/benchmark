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

    def forward(self, x229, x237):
        x238=operator.add(x229, x237)
        return x238

m = M().eval()
x229 = torch.randn(torch.Size([1, 432, 14, 14]))
x237 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x229, x237)
end = time.time()
print(end-start)
