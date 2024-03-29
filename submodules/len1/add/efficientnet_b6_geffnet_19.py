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

    def forward(self, x366, x352):
        x367=operator.add(x366, x352)
        return x367

m = M().eval()
x366 = torch.randn(torch.Size([1, 200, 14, 14]))
x352 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x366, x352)
end = time.time()
print(end-start)
