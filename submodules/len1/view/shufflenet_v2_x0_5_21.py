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

    def forward(self, x249, x242, x244, x245):
        x250=x249.view(x242, -1, x244, x245)
        return x250

m = M().eval()
x249 = torch.randn(torch.Size([1, 48, 2, 14, 14]))
x242 = 1
x244 = 14
x245 = 14
start = time.time()
output = m(x249, x242, x244, x245)
end = time.time()
print(end-start)
