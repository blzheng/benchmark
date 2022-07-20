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

    def forward(self, x958, x943, x974):
        x959=operator.add(x958, x943)
        x975=operator.add(x974, x959)
        return x975

m = M().eval()
x958 = torch.randn(torch.Size([1, 384, 7, 7]))
x943 = torch.randn(torch.Size([1, 384, 7, 7]))
x974 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x958, x943, x974)
end = time.time()
print(end-start)
