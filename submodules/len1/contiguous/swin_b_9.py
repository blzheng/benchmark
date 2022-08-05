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

    def forward(self, x232):
        x233=x232.contiguous()
        return x233

m = M().eval()
x232 = torch.randn(torch.Size([16, 49, 49]))
start = time.time()
output = m(x232)
end = time.time()
print(end-start)
