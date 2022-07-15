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

    def forward(self, x622, x607):
        x623=operator.add(x622, x607)
        return x623

m = M().eval()
x622 = torch.randn(torch.Size([1, 384, 7, 7]))
x607 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x622, x607)
end = time.time()
print(end-start)
