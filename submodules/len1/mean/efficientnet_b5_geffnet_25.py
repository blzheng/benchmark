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

    def forward(self, x373):
        x374=x373.mean((2, 3),keepdim=True)
        return x374

m = M().eval()
x373 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x373)
end = time.time()
print(end-start)
