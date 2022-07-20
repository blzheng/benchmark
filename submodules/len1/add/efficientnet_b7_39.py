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

    def forward(self, x714, x699):
        x715=operator.add(x714, x699)
        return x715

m = M().eval()
x714 = torch.randn(torch.Size([1, 384, 7, 7]))
x699 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x714, x699)
end = time.time()
print(end-start)