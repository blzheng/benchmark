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

    def forward(self, x240, x242, x246, x244, x245):
        x247=x240.view(x242, 2, x246, x244, x245)
        return x247

m = M().eval()
x240 = torch.randn(torch.Size([1, 232, 14, 14]))
x242 = 1
x246 = 116
x244 = 14
x245 = 14
start = time.time()
output = m(x240, x242, x246, x244, x245)
end = time.time()
print(end-start)
