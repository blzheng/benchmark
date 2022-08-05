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

    def forward(self, x562):
        x563=x562.contiguous()
        return x563

m = M().eval()
x562 = torch.randn(torch.Size([24, 49, 49]))
start = time.time()
output = m(x562)
end = time.time()
print(end-start)
