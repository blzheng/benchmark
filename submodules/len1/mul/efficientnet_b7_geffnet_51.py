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

    def forward(self, x760, x765):
        x766=operator.mul(x760, x765)
        return x766

m = M().eval()
x760 = torch.randn(torch.Size([1, 2304, 7, 7]))
x765 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x760, x765)
end = time.time()
print(end-start)
