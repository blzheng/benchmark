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

    def forward(self, x204, x190, x219):
        x205=operator.add(x204, x190)
        x220=operator.add(x219, x205)
        return x220

m = M().eval()
x204 = torch.randn(torch.Size([1, 192, 7, 7]))
x190 = torch.randn(torch.Size([1, 192, 7, 7]))
x219 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x204, x190, x219)
end = time.time()
print(end-start)
