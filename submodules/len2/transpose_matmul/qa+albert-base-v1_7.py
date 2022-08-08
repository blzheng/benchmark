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

    def forward(self, x301, x296):
        x307=x301.transpose(-1, -2)
        x308=torch.matmul(x296, x307)
        return x308

m = M().eval()
x301 = torch.randn(torch.Size([1, 12, 384, 64]))
x296 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x301, x296)
end = time.time()
print(end-start)
