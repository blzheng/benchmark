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

    def forward(self, x372, x383):
        x384=x372.transpose(-1, -2)
        x385=torch.matmul(x383, x384)
        return x385

m = M().eval()
x372 = torch.randn(torch.Size([1, 4, 384, 64]))
x383 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x372, x383)
end = time.time()
print(end-start)
