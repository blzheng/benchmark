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

    def forward(self, x378, x374):
        x379=x378.sigmoid()
        x380=operator.mul(x374, x379)
        return x380

m = M().eval()
x378 = torch.randn(torch.Size([1, 1632, 1, 1]))
x374 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x378, x374)
end = time.time()
print(end-start)
