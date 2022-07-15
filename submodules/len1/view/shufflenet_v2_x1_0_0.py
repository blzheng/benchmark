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

    def forward(self, x20, x24, x22, x23):
        x25=x18.view(x20, 2, x24, x22, x23)
        return x25

m = M().eval()
x20 = 1
x24 = 58
x22 = 28
x23 = 28
start = time.time()
output = m(x20, x24, x22, x23)
end = time.time()
print(end-start)
