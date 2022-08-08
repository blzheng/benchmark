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

    def forward(self, x412, x407):
        x418=x412.transpose(-1, -2)
        x419=torch.matmul(x407, x418)
        return x419

m = M().eval()
x412 = torch.randn(torch.Size([1, 12, 384, 64]))
x407 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x412, x407)
end = time.time()
print(end-start)
