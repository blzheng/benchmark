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

    def forward(self, x438):
        x439=x438.permute(2, 0, 1)
        return x439

m = M().eval()
x438 = torch.randn(torch.Size([49, 49, 16]))
start = time.time()
output = m(x438)
end = time.time()
print(end-start)
