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

    def forward(self, x286):
        x287=x286.permute(0, 2, 1, 3)
        return x287

m = M().eval()
x286 = torch.randn(torch.Size([1, 384, 12, 64]))
start = time.time()
output = m(x286)
end = time.time()
print(end-start)
