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

    def forward(self, x390, x385):
        x391=operator.mul(x390, x385)
        return x391

m = M().eval()
x390 = torch.randn(torch.Size([1, 1536, 1, 1]))
x385 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x390, x385)
end = time.time()
print(end-start)
