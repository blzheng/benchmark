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

    def forward(self, x248, x234):
        x249=operator.add(x248, x234)
        return x249

m = M().eval()
x248 = torch.randn(torch.Size([1, 136, 14, 14]))
x234 = torch.randn(torch.Size([1, 136, 14, 14]))
start = time.time()
output = m(x248, x234)
end = time.time()
print(end-start)
