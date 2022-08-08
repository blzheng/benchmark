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

    def forward(self, x498, x509):
        x510=x498.transpose(-1, -2)
        x511=torch.matmul(x509, x510)
        return x511

m = M().eval()
x498 = torch.randn(torch.Size([1, 4, 384, 64]))
x509 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x498, x509)
end = time.time()
print(end-start)
