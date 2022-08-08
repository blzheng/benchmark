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

    def forward(self, x42):
        x48=x42.transpose(-1, -2)
        return x48

m = M().eval()
x42 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x42)
end = time.time()
print(end-start)
