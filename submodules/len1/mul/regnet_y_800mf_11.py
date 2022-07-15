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

    def forward(self, x196, x191):
        x197=operator.mul(x196, x191)
        return x197

m = M().eval()
x196 = torch.randn(torch.Size([1, 320, 1, 1]))
x191 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x196, x191)
end = time.time()
print(end-start)
