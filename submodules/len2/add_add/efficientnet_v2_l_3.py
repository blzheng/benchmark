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

    def forward(self, x258, x243, x274):
        x259=operator.add(x258, x243)
        x275=operator.add(x274, x259)
        return x275

m = M().eval()
x258 = torch.randn(torch.Size([1, 192, 14, 14]))
x243 = torch.randn(torch.Size([1, 192, 14, 14]))
x274 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x258, x243, x274)
end = time.time()
print(end-start)
