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

    def forward(self, x48, x42):
        x49=operator.add(x48, x42)
        return x49

m = M().eval()
x48 = torch.randn(torch.Size([1, 64, 56, 56]))
x42 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x48, x42)
end = time.time()
print(end-start)
