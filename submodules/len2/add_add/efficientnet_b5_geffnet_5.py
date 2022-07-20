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

    def forward(self, x515, x501, x530):
        x516=operator.add(x515, x501)
        x531=operator.add(x530, x516)
        return x531

m = M().eval()
x515 = torch.randn(torch.Size([1, 304, 7, 7]))
x501 = torch.randn(torch.Size([1, 304, 7, 7]))
x530 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x515, x501, x530)
end = time.time()
print(end-start)
