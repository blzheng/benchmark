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

    def forward(self, x612, x617):
        x618=operator.mul(x612, x617)
        return x618

m = M().eval()
x612 = torch.randn(torch.Size([1, 2064, 7, 7]))
x617 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x612, x617)
end = time.time()
print(end-start)
