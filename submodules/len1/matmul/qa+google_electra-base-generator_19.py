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

    def forward(self, x432, x420):
        x433=torch.matmul(x432, x420)
        return x433

m = M().eval()
x432 = torch.randn(torch.Size([1, 4, 384, 384]))
x420 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x432, x420)
end = time.time()
print(end-start)
