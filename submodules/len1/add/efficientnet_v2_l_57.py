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

    def forward(self, x830, x815):
        x831=operator.add(x830, x815)
        return x831

m = M().eval()
x830 = torch.randn(torch.Size([1, 384, 7, 7]))
x815 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x830, x815)
end = time.time()
print(end-start)
