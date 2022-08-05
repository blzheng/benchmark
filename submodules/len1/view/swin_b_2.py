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

    def forward(self, x61):
        x62=x61.view(49, 49, -1)
        return x62

m = M().eval()
x61 = torch.randn(torch.Size([2401, 8]))
start = time.time()
output = m(x61)
end = time.time()
print(end-start)
