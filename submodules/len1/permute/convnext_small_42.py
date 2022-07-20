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

    def forward(self, x218):
        x219=torch.permute(x218, [0, 2, 3, 1])
        return x219

m = M().eval()
x218 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x218)
end = time.time()
print(end-start)
