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

    def forward(self, x463, x448, x479):
        x464=operator.add(x463, x448)
        x480=operator.add(x479, x464)
        return x480

m = M().eval()
x463 = torch.randn(torch.Size([1, 200, 14, 14]))
x448 = torch.randn(torch.Size([1, 200, 14, 14]))
x479 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x463, x448, x479)
end = time.time()
print(end-start)
