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

    def forward(self, x111):
        x114=operator.floordiv(x111, 2)
        return x114

m = M().eval()
x111 = 488
start = time.time()
output = m(x111)
end = time.time()
print(end-start)