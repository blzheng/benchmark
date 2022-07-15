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

    def forward(self, x133):
        x136=operator.floordiv(x133, 2)
        return x136

m = M().eval()
x133 = 488
start = time.time()
output = m(x133)
end = time.time()
print(end-start)
