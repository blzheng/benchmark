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

    def forward(self, x30, x32):
        x33=operator.add(x30, x32)
        return x33

m = M().eval()
x30 = torch.randn(torch.Size([1, 128, 28, 28]))
x32 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x30, x32)
end = time.time()
print(end-start)
