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

    def forward(self, x272, x257):
        x273=operator.add(x272, x257)
        return x273

m = M().eval()
x272 = torch.randn(torch.Size([1, 80, 28, 28]))
x257 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x272, x257)
end = time.time()
print(end-start)
