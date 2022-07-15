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

    def forward(self, x477, x482):
        x483=operator.mul(x477, x482)
        return x483

m = M().eval()
x477 = torch.randn(torch.Size([1, 1824, 7, 7]))
x482 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x477, x482)
end = time.time()
print(end-start)
