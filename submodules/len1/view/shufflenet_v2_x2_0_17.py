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

    def forward(self, x198, x200, x201):
        x206=x205.view(x198, -1, x200, x201)
        return x206

m = M().eval()
x198 = 1
x200 = 14
x201 = 14
start = time.time()
output = m(x198, x200, x201)
end = time.time()
print(end-start)
