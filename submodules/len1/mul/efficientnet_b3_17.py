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

    def forward(self, x270, x265):
        x271=operator.mul(x270, x265)
        return x271

m = M().eval()
x270 = torch.randn(torch.Size([1, 816, 1, 1]))
x265 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x270, x265)
end = time.time()
print(end-start)
