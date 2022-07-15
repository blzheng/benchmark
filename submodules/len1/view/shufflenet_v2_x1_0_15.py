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

    def forward(self, x176, x178, x179):
        x184=x183.view(x176, -1, x178, x179)
        return x184

m = M().eval()
x176 = 1
x178 = 14
x179 = 14
start = time.time()
output = m(x176, x178, x179)
end = time.time()
print(end-start)
