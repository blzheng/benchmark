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

    def forward(self, x101, x99):
        x102=operator.add(x101, (256,))
        x103=x99.view(x102)
        return x103

m = M().eval()
x101 = (1, 384, )
x99 = torch.randn(torch.Size([1, 384, 4, 64]))
start = time.time()
output = m(x101, x99)
end = time.time()
print(end-start)
