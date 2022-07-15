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

    def forward(self, x678, x673):
        x679=operator.mul(x678, x673)
        return x679

m = M().eval()
x678 = torch.randn(torch.Size([1, 2304, 1, 1]))
x673 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x678, x673)
end = time.time()
print(end-start)
