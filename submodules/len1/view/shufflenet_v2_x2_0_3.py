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

    def forward(self, x49, x42, x44, x45):
        x50=x49.view(x42, -1, x44, x45)
        return x50

m = M().eval()
x49 = torch.randn(torch.Size([1, 122, 2, 28, 28]))
x42 = 1
x44 = 28
x45 = 28
start = time.time()
output = m(x49, x42, x44, x45)
end = time.time()
print(end-start)
