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

    def forward(self, x83, x77):
        x84=operator.add(x83, x77)
        return x84

m = M().eval()
x83 = torch.randn(torch.Size([1, 80, 28, 28]))
x77 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x83, x77)
end = time.time()
print(end-start)
