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

    def forward(self, x327, x325):
        x328=operator.add(x327, (4, 64))
        x329=x325.view(x328)
        return x329

m = M().eval()
x327 = (1, 384, )
x325 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x327, x325)
end = time.time()
print(end-start)
