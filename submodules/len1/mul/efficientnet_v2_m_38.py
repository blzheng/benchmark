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

    def forward(self, x697, x692):
        x698=operator.mul(x697, x692)
        return x698

m = M().eval()
x697 = torch.randn(torch.Size([1, 1824, 1, 1]))
x692 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x697, x692)
end = time.time()
print(end-start)
