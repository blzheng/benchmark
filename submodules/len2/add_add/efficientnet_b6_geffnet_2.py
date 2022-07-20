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

    def forward(self, x203, x189, x218):
        x204=operator.add(x203, x189)
        x219=operator.add(x218, x204)
        return x219

m = M().eval()
x203 = torch.randn(torch.Size([1, 72, 28, 28]))
x189 = torch.randn(torch.Size([1, 72, 28, 28]))
x218 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x203, x189, x218)
end = time.time()
print(end-start)