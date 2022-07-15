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

    def forward(self, x189, x184):
        x190=operator.mul(x189, x184)
        return x190

m = M().eval()
x189 = torch.randn(torch.Size([1, 640, 1, 1]))
x184 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x189, x184)
end = time.time()
print(end-start)
