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

    def forward(self, x184, x180):
        x185=x184.sigmoid()
        x186=operator.mul(x180, x185)
        return x186

m = M().eval()
x184 = torch.randn(torch.Size([1, 432, 1, 1]))
x180 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x184, x180)
end = time.time()
print(end-start)
