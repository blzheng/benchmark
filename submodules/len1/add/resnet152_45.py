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

    def forward(self, x468, x460):
        x469=operator.add(x468, x460)
        return x469

m = M().eval()
x468 = torch.randn(torch.Size([1, 1024, 14, 14]))
x460 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x468, x460)
end = time.time()
print(end-start)
