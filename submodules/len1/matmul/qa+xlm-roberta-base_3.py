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

    def forward(self, x95, x83):
        x96=torch.matmul(x95, x83)
        return x96

m = M().eval()
x95 = torch.randn(torch.Size([1, 12, 384, 384]))
x83 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x95, x83)
end = time.time()
print(end-start)
