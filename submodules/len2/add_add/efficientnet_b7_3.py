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

    def forward(self, x414, x399, x430):
        x415=operator.add(x414, x399)
        x431=operator.add(x430, x415)
        return x431

m = M().eval()
x414 = torch.randn(torch.Size([1, 160, 14, 14]))
x399 = torch.randn(torch.Size([1, 160, 14, 14]))
x430 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x414, x399, x430)
end = time.time()
print(end-start)
