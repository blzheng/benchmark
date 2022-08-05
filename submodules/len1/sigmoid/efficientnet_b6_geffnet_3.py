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

    def forward(self, x51):
        x52=x51.sigmoid()
        return x52

m = M().eval()
x51 = torch.randn(torch.Size([1, 192, 1, 1]))
start = time.time()
output = m(x51)
end = time.time()
print(end-start)
