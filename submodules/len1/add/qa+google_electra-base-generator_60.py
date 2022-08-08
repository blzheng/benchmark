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

    def forward(self, x380):
        x381=operator.add(x380, (4, 64))
        return x381

m = M().eval()
x380 = (1, 384, )
start = time.time()
output = m(x380)
end = time.time()
print(end-start)
