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

    def forward(self, x14):
        x15=torch.flatten(x14, 1)
        return x15

m = M().eval()
x14 = torch.randn(torch.Size([1, 256, 6, 6]))
start = time.time()
output = m(x14)
end = time.time()
print(end-start)
