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

    def forward(self, x319, x315):
        x320=x319.sigmoid()
        x321=operator.mul(x315, x320)
        return x321

m = M().eval()
x319 = torch.randn(torch.Size([1, 960, 1, 1]))
x315 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x319, x315)
end = time.time()
print(end-start)
