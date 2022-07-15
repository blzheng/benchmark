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

    def forward(self, x552, x557):
        x558=operator.mul(x552, x557)
        return x558

m = M().eval()
x552 = torch.randn(torch.Size([1, 2064, 7, 7]))
x557 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x552, x557)
end = time.time()
print(end-start)
