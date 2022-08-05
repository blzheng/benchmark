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

    def forward(self, x218, x225):
        x226=operator.add(x218, x225)
        return x226

m = M().eval()
x218 = torch.randn(torch.Size([1, 14, 14, 384]))
x225 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x218, x225)
end = time.time()
print(end-start)
