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

    def forward(self, x360):
        x361=x360.contiguous()
        return x361

m = M().eval()
x360 = torch.randn(torch.Size([1, 232, 2, 7, 7]))
start = time.time()
output = m(x360)
end = time.time()
print(end-start)
