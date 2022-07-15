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

    def forward(self, x575, x561):
        x576=operator.add(x575, x561)
        return x576

m = M().eval()
x575 = torch.randn(torch.Size([1, 344, 7, 7]))
x561 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x575, x561)
end = time.time()
print(end-start)
