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

    def forward(self, x108, x104):
        x109=operator.add(x108, (12, 64))
        x110=x104.view(x109)
        return x110

m = M().eval()
x108 = (1, 384, )
x104 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x108, x104)
end = time.time()
print(end-start)
