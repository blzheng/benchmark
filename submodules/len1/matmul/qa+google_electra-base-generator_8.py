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

    def forward(self, x215, x216):
        x217=torch.matmul(x215, x216)
        return x217

m = M().eval()
x215 = torch.randn(torch.Size([1, 4, 384, 64]))
x216 = torch.randn(torch.Size([1, 4, 64, 384]))
start = time.time()
output = m(x215, x216)
end = time.time()
print(end-start)
