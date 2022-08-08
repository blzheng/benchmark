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

    def forward(self, x106, x119):
        x120=x106.view(x119)
        return x120

m = M().eval()
x106 = torch.randn(torch.Size([1, 384, 768]))
x119 = (1, 384, 12, 64, )
start = time.time()
output = m(x106, x119)
end = time.time()
print(end-start)
