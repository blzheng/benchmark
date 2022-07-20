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

    def forward(self, x56, x42, x71):
        x57=operator.add(x56, x42)
        x72=operator.add(x71, x57)
        return x72

m = M().eval()
x56 = torch.randn(torch.Size([1, 24, 56, 56]))
x42 = torch.randn(torch.Size([1, 24, 56, 56]))
x71 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x56, x42, x71)
end = time.time()
print(end-start)
