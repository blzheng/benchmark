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

    def forward(self, x646, x641):
        x647=operator.mul(x646, x641)
        return x647

m = M().eval()
x646 = torch.randn(torch.Size([1, 2304, 1, 1]))
x641 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x646, x641)
end = time.time()
print(end-start)
