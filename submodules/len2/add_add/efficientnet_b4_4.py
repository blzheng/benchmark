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

    def forward(self, x448, x433, x464):
        x449=operator.add(x448, x433)
        x465=operator.add(x464, x449)
        return x465

m = M().eval()
x448 = torch.randn(torch.Size([1, 272, 7, 7]))
x433 = torch.randn(torch.Size([1, 272, 7, 7]))
x464 = torch.randn(torch.Size([1, 272, 7, 7]))
start = time.time()
output = m(x448, x433, x464)
end = time.time()
print(end-start)
