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

    def forward(self, x1020, x1005):
        x1021=operator.add(x1020, x1005)
        return x1021

m = M().eval()
x1020 = torch.randn(torch.Size([1, 640, 7, 7]))
x1005 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x1020, x1005)
end = time.time()
print(end-start)
