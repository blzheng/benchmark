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

    def forward(self, x141, x144):
        x145=x141.view(x144)
        return x145

m = M().eval()
x141 = torch.randn(torch.Size([1, 384, 4, 64]))
x144 = (1, 384, 256, )
start = time.time()
output = m(x141, x144)
end = time.time()
print(end-start)
