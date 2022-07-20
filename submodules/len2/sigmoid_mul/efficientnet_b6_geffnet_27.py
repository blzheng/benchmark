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

    def forward(self, x407, x403):
        x408=x407.sigmoid()
        x409=operator.mul(x403, x408)
        return x409

m = M().eval()
x407 = torch.randn(torch.Size([1, 1200, 1, 1]))
x403 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x407, x403)
end = time.time()
print(end-start)
