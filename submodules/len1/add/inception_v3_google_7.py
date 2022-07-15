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

    def forward(self, x113):
        x141=torch._C._nn.avg_pool2d(x113,kernel_size=3, stride=1, padding=1)
        return x141

m = M().eval()
x113 = torch.randn(torch.Size([1, 768, 12, 12]))
start = time.time()
output = m(x113)
end = time.time()
print(end-start)
