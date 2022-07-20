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

    def forward(self, x173, x159, x188):
        x174=operator.add(x173, x159)
        x189=operator.add(x188, x174)
        return x189

m = M().eval()
x173 = torch.randn(torch.Size([1, 64, 28, 28]))
x159 = torch.randn(torch.Size([1, 64, 28, 28]))
x188 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x173, x159, x188)
end = time.time()
print(end-start)
