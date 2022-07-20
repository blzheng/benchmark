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

    def forward(self, x840, x825, x856):
        x841=operator.add(x840, x825)
        x857=operator.add(x856, x841)
        return x857

m = M().eval()
x840 = torch.randn(torch.Size([1, 640, 7, 7]))
x825 = torch.randn(torch.Size([1, 640, 7, 7]))
x856 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x840, x825, x856)
end = time.time()
print(end-start)
