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

    def forward(self, x308, x310, x314, x312, x313):
        x315=x308.view(x310, 2, x314, x312, x313)
        return x315

m = M().eval()
x308 = torch.randn(torch.Size([1, 976, 7, 7]))
x310 = 1
x314 = 488
x312 = 7
x313 = 7
start = time.time()
output = m(x308, x310, x314, x312, x313)
end = time.time()
print(end-start)
