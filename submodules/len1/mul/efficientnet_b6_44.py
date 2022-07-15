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

    def forward(self, x695, x690):
        x696=operator.mul(x695, x690)
        return x696

m = M().eval()
x695 = torch.randn(torch.Size([1, 3456, 1, 1]))
x690 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x695, x690)
end = time.time()
print(end-start)
