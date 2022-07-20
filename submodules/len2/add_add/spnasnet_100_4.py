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

    def forward(self, x184, x175, x194):
        x185=operator.add(x184, x175)
        x195=operator.add(x194, x185)
        return x195

m = M().eval()
x184 = torch.randn(torch.Size([1, 192, 7, 7]))
x175 = torch.randn(torch.Size([1, 192, 7, 7]))
x194 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x184, x175, x194)
end = time.time()
print(end-start)
