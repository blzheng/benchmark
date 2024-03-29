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

    def forward(self, x82):
        x83=x82.sigmoid()
        return x83

m = M().eval()
x82 = torch.randn(torch.Size([1, 192, 1, 1]))
start = time.time()
output = m(x82)
end = time.time()
print(end-start)
