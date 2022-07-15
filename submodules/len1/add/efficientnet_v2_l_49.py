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

    def forward(self, x702, x687):
        x703=operator.add(x702, x687)
        return x703

m = M().eval()
x702 = torch.randn(torch.Size([1, 384, 7, 7]))
x687 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x702, x687)
end = time.time()
print(end-start)
