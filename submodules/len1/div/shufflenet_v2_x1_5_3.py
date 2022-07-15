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

    def forward(self, x87):
        x90=operator.floordiv(x87, 2)
        return x90

m = M().eval()
x87 = 176
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
