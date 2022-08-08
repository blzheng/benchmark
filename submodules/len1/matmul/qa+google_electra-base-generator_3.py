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

    def forward(self, x96, x84):
        x97=torch.matmul(x96, x84)
        return x97

m = M().eval()
x96 = torch.randn(torch.Size([1, 4, 384, 384]))
x84 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x96, x84)
end = time.time()
print(end-start)
