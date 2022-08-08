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

    def forward(self, x310, x308):
        x311=operator.add(x310, (768,))
        x312=x308.view(x311)
        return x312

m = M().eval()
x310 = (1, 384, )
x308 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x310, x308)
end = time.time()
print(end-start)
