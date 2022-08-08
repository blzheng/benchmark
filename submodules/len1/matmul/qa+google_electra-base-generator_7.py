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

    def forward(self, x180, x168):
        x181=torch.matmul(x180, x168)
        return x181

m = M().eval()
x180 = torch.randn(torch.Size([1, 4, 384, 384]))
x168 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x180, x168)
end = time.time()
print(end-start)
