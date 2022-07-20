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

    def forward(self, x377, x373):
        x378=x377.sigmoid()
        x379=operator.mul(x373, x378)
        return x379

m = M().eval()
x377 = torch.randn(torch.Size([1, 2304, 1, 1]))
x373 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x377, x373)
end = time.time()
print(end-start)
