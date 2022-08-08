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

    def forward(self, x455, x466):
        x467=x455.transpose(-1, -2)
        x468=torch.matmul(x466, x467)
        return x468

m = M().eval()
x455 = torch.randn(torch.Size([1, 12, 384, 64]))
x466 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x455, x466)
end = time.time()
print(end-start)
