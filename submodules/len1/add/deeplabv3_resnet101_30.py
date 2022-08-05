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

    def forward(self, x320, x322):
        x323=operator.add(x320, x322)
        return x323

m = M().eval()
x320 = torch.randn(torch.Size([1, 2048, 28, 28]))
x322 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x320, x322)
end = time.time()
print(end-start)
