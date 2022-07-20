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

    def forward(self, x644, x640):
        x645=x644.sigmoid()
        x646=operator.mul(x640, x645)
        return x646

m = M().eval()
x644 = torch.randn(torch.Size([1, 2304, 1, 1]))
x640 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x644, x640)
end = time.time()
print(end-start)
