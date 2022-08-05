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

    def forward(self, x379, x386):
        x387=operator.add(x379, x386)
        return x387

m = M().eval()
x379 = torch.randn(torch.Size([1, 14, 14, 512]))
x386 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x379, x386)
end = time.time()
print(end-start)
