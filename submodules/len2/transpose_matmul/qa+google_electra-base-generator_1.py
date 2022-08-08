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

    def forward(self, x78, x89):
        x90=x78.transpose(-1, -2)
        x91=torch.matmul(x89, x90)
        return x91

m = M().eval()
x78 = torch.randn(torch.Size([1, 4, 384, 64]))
x89 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x78, x89)
end = time.time()
print(end-start)
