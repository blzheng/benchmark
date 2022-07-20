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

    def forward(self, x3, x8, x12):
        x4=operator.add(x3, -0.030000000000000027)
        x13=torch.cat((x4, x8, x12), 1)
        return x13

m = M().eval()
x3 = torch.randn(torch.Size([1, 1, 224, 224]))
x8 = torch.randn(torch.Size([1, 1, 224, 224]))
x12 = torch.randn(torch.Size([1, 1, 224, 224]))
start = time.time()
output = m(x3, x8, x12)
end = time.time()
print(end-start)
