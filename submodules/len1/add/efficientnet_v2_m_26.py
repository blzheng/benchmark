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

    def forward(self, x367, x352):
        x368=operator.add(x367, x352)
        return x368

m = M().eval()
x367 = torch.randn(torch.Size([1, 176, 14, 14]))
x352 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x367, x352)
end = time.time()
print(end-start)
