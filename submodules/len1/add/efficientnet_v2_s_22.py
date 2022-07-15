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

    def forward(self, x346, x331):
        x347=operator.add(x346, x331)
        return x347

m = M().eval()
x346 = torch.randn(torch.Size([1, 256, 7, 7]))
x331 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x346, x331)
end = time.time()
print(end-start)
