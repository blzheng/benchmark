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

    def forward(self, x162, x173):
        x174=x162.transpose(-1, -2)
        x175=torch.matmul(x173, x174)
        return x175

m = M().eval()
x162 = torch.randn(torch.Size([1, 4, 384, 64]))
x173 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x162, x173)
end = time.time()
print(end-start)
