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

    def forward(self, x450, x446):
        x451=x450.sigmoid()
        x452=operator.mul(x446, x451)
        return x452

m = M().eval()
x450 = torch.randn(torch.Size([1, 1344, 1, 1]))
x446 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x450, x446)
end = time.time()
print(end-start)