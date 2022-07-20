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

    def forward(self, x26):
        x27=x26.contiguous()
        return x27

m = M().eval()
x26 = torch.randn(torch.Size([1, 58, 2, 28, 28]))
start = time.time()
output = m(x26)
end = time.time()
print(end-start)
