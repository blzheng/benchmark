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

    def forward(self, x218, x204, x233):
        x219=operator.add(x218, x204)
        x234=operator.add(x233, x219)
        return x234

m = M().eval()
x218 = torch.randn(torch.Size([1, 112, 14, 14]))
x204 = torch.randn(torch.Size([1, 112, 14, 14]))
x233 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x218, x204, x233)
end = time.time()
print(end-start)
