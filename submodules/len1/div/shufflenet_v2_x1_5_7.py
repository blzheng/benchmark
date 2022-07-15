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

    def forward(self, x177):
        x180=operator.floordiv(x177, 2)
        return x180

m = M().eval()
x177 = 352
start = time.time()
output = m(x177)
end = time.time()
print(end-start)
