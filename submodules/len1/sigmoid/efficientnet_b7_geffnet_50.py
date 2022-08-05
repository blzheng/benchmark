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

    def forward(self, x749):
        x750=x749.sigmoid()
        return x750

m = M().eval()
x749 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x749)
end = time.time()
print(end-start)
