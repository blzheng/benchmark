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

    def forward(self, x218, x220, x224, x222, x223):
        x225=x218.view(x220, 2, x224, x222, x223)
        return x225

m = M().eval()
x218 = torch.randn(torch.Size([1, 352, 14, 14]))
x220 = 1
x224 = 176
x222 = 14
x223 = 14
start = time.time()
output = m(x218, x220, x224, x222, x223)
end = time.time()
print(end-start)
