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

    def forward(self, x211, x219):
        x220=operator.add(x211, x219)
        return x220

m = M().eval()
x211 = torch.randn(torch.Size([1, 400, 7, 7]))
x219 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x211, x219)
end = time.time()
print(end-start)
