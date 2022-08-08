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

    def forward(self, x483, x449):
        x484=operator.add(x483, x449)
        return x484

m = M().eval()
x483 = torch.randn(torch.Size([1, 384, 256]))
x449 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x483, x449)
end = time.time()
print(end-start)
