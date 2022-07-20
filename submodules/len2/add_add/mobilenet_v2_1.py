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

    def forward(self, x68, x76, x85):
        x77=operator.add(x68, x76)
        x86=operator.add(x77, x85)
        return x86

m = M().eval()
x68 = torch.randn(torch.Size([1, 64, 14, 14]))
x76 = torch.randn(torch.Size([1, 64, 14, 14]))
x85 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x68, x76, x85)
end = time.time()
print(end-start)
