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

    def forward(self, x485, x471):
        x486=operator.add(x485, x471)
        return x486

m = M().eval()
x485 = torch.randn(torch.Size([1, 344, 7, 7]))
x471 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x485, x471)
end = time.time()
print(end-start)