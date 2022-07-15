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

    def forward(self, x303, x288):
        x304=operator.add(x303, x288)
        return x304

m = M().eval()
x303 = torch.randn(torch.Size([1, 176, 14, 14]))
x288 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x303, x288)
end = time.time()
print(end-start)
