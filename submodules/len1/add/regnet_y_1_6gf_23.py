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

    def forward(self, x377, x391):
        x392=operator.add(x377, x391)
        return x392

m = M().eval()
x377 = torch.randn(torch.Size([1, 336, 14, 14]))
x391 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x377, x391)
end = time.time()
print(end-start)
