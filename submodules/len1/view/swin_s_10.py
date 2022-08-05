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

    def forward(self, x253):
        x254=x253.view(49, 49, -1)
        return x254

m = M().eval()
x253 = torch.randn(torch.Size([2401, 12]))
start = time.time()
output = m(x253)
end = time.time()
print(end-start)
