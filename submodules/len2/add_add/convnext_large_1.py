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

    def forward(self, x66, x56, x77):
        x67=operator.add(x66, x56)
        x78=operator.add(x77, x67)
        return x78

m = M().eval()
x66 = torch.randn(torch.Size([1, 384, 28, 28]))
x56 = torch.randn(torch.Size([1, 384, 28, 28]))
x77 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x66, x56, x77)
end = time.time()
print(end-start)
