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

    def forward(self, x329, x343):
        x344=operator.add(x329, x343)
        return x344

m = M().eval()
x329 = torch.randn(torch.Size([1, 2904, 14, 14]))
x343 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x329, x343)
end = time.time()
print(end-start)
