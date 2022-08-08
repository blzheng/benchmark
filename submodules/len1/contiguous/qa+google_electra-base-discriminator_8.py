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

    def forward(self, x391):
        x392=x391.contiguous()
        return x392

m = M().eval()
x391 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x391)
end = time.time()
print(end-start)
