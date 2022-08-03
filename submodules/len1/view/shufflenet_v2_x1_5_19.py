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

    def forward(self, x227, x220, x222, x223):
        x228=x227.view(x220, -1, x222, x223)
        return x228

m = M().eval()
x227 = torch.randn(torch.Size([1, 176, 2, 14, 14]))
x220 = 1
x222 = 14
x223 = 14
start = time.time()
output = m(x227, x220, x222, x223)
end = time.time()
print(end-start)
