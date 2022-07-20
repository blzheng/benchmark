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

    def forward(self, x540, x536):
        x541=x540.sigmoid()
        x542=operator.mul(x536, x541)
        return x542

m = M().eval()
x540 = torch.randn(torch.Size([1, 1344, 1, 1]))
x536 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x540, x536)
end = time.time()
print(end-start)
