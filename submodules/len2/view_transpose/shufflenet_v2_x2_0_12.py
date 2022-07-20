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

    def forward(self, x286, x288, x292, x290, x291):
        x293=x286.view(x288, 2, x292, x290, x291)
        x294=torch.transpose(x293, 1, 2)
        return x294

m = M().eval()
x286 = torch.randn(torch.Size([1, 976, 7, 7]))
x288 = 1
x292 = 488
x290 = 7
x291 = 7
start = time.time()
output = m(x286, x288, x292, x290, x291)
end = time.time()
print(end-start)
