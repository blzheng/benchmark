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

    def forward(self, x23, x37):
        x38=operator.add(x23, x37)
        return x38

m = M().eval()
x23 = torch.randn(torch.Size([1, 144, 28, 28]))
x37 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x23, x37)
end = time.time()
print(end-start)
