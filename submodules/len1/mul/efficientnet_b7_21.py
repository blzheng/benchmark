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

    def forward(self, x330, x325):
        x331=operator.mul(x330, x325)
        return x331

m = M().eval()
x330 = torch.randn(torch.Size([1, 960, 1, 1]))
x325 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x330, x325)
end = time.time()
print(end-start)
