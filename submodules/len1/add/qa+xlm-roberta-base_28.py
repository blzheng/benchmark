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

    def forward(self, x188, x154):
        x189=operator.add(x188, x154)
        return x189

m = M().eval()
x188 = torch.randn(torch.Size([1, 384, 768]))
x154 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x188, x154)
end = time.time()
print(end-start)
