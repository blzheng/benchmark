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

    def forward(self, x465, x461):
        x466=x465.sigmoid()
        x467=operator.mul(x461, x466)
        return x467

m = M().eval()
x465 = torch.randn(torch.Size([1, 1344, 1, 1]))
x461 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x465, x461)
end = time.time()
print(end-start)
