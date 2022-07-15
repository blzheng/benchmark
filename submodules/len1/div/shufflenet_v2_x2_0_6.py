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

    def forward(self, x155):
        x158=operator.floordiv(x155, 2)
        return x158

m = M().eval()
x155 = 488
start = time.time()
output = m(x155)
end = time.time()
print(end-start)
