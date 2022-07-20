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

    def forward(self, x683, x668, x699):
        x684=operator.add(x683, x668)
        x700=operator.add(x699, x684)
        return x700

m = M().eval()
x683 = torch.randn(torch.Size([1, 576, 7, 7]))
x668 = torch.randn(torch.Size([1, 576, 7, 7]))
x699 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x683, x668, x699)
end = time.time()
print(end-start)
