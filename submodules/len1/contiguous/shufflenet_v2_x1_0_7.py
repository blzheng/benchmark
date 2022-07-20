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

    def forward(self, x182):
        x183=x182.contiguous()
        return x183

m = M().eval()
x182 = torch.randn(torch.Size([1, 116, 2, 14, 14]))
start = time.time()
output = m(x182)
end = time.time()
print(end-start)
