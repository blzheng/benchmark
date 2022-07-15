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

    def forward(self, x323, x309):
        x324=operator.add(x323, x309)
        return x324

m = M().eval()
x323 = torch.randn(torch.Size([1, 160, 14, 14]))
x309 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x323, x309)
end = time.time()
print(end-start)
