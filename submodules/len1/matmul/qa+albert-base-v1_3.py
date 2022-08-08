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

    def forward(self, x91, x84):
        x92=torch.matmul(x91, x84)
        return x92

m = M().eval()
x91 = torch.randn(torch.Size([1, 12, 384, 384]))
x84 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x91, x84)
end = time.time()
print(end-start)
