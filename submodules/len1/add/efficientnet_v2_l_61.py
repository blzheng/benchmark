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

    def forward(self, x894, x879):
        x895=operator.add(x894, x879)
        return x895

m = M().eval()
x894 = torch.randn(torch.Size([1, 384, 7, 7]))
x879 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x894, x879)
end = time.time()
print(end-start)
