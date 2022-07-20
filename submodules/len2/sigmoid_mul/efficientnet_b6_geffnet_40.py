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

    def forward(self, x601, x597):
        x602=x601.sigmoid()
        x603=operator.mul(x597, x602)
        return x603

m = M().eval()
x601 = torch.randn(torch.Size([1, 2064, 1, 1]))
x597 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x601, x597)
end = time.time()
print(end-start)
