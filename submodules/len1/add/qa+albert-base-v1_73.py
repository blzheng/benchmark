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

    def forward(self, x471, x468):
        x472=operator.add(x471, x468)
        return x472

m = M().eval()
x471 = torch.randn(torch.Size([1, 384, 768]))
x468 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x471, x468)
end = time.time()
print(end-start)
