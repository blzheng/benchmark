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

    def forward(self, x265):
        x268=operator.floordiv(x265, 2)
        return x268

m = M().eval()
x265 = 352
start = time.time()
output = m(x265)
end = time.time()
print(end-start)
