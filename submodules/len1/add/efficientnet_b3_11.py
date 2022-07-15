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

    def forward(self, x258, x243):
        x259=operator.add(x258, x243)
        return x259

m = M().eval()
x258 = torch.randn(torch.Size([1, 136, 14, 14]))
x243 = torch.randn(torch.Size([1, 136, 14, 14]))
start = time.time()
output = m(x258, x243)
end = time.time()
print(end-start)
