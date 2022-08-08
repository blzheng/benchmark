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

    def forward(self, x379):
        x380=operator.add(x379, (12, 64))
        return x380

m = M().eval()
x379 = (1, 384, )
start = time.time()
output = m(x379)
end = time.time()
print(end-start)
