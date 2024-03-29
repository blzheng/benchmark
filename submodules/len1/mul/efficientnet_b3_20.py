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

    def forward(self, x316, x311):
        x317=operator.mul(x316, x311)
        return x317

m = M().eval()
x316 = torch.randn(torch.Size([1, 1392, 1, 1]))
x311 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x316, x311)
end = time.time()
print(end-start)
