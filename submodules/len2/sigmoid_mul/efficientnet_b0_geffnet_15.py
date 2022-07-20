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

    def forward(self, x230, x226):
        x231=x230.sigmoid()
        x232=operator.mul(x226, x231)
        return x232

m = M().eval()
x230 = torch.randn(torch.Size([1, 1152, 1, 1]))
x226 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x230, x226)
end = time.time()
print(end-start)
