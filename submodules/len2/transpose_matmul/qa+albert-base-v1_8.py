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

    def forward(self, x338, x333):
        x344=x338.transpose(-1, -2)
        x345=torch.matmul(x333, x344)
        return x345

m = M().eval()
x338 = torch.randn(torch.Size([1, 12, 384, 64]))
x333 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x338, x333)
end = time.time()
print(end-start)
