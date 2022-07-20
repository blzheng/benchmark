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

    def forward(self, x35, x31):
        x36=x35.sigmoid()
        x37=operator.mul(x31, x36)
        return x37

m = M().eval()
x35 = torch.randn(torch.Size([1, 24, 1, 1]))
x31 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x35, x31)
end = time.time()
print(end-start)
