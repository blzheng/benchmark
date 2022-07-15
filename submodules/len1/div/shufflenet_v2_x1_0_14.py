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

    def forward(self, x333):
        x336=operator.floordiv(x333, 2)
        return x336

m = M().eval()
x333 = 464
start = time.time()
output = m(x333)
end = time.time()
print(end-start)
