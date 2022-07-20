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
        self.relu63 = ReLU(inplace=True)

    def forward(self, x209, x217):
        x218=operator.add(x209, x217)
        x219=self.relu63(x218)
        return x219

m = M().eval()
x209 = torch.randn(torch.Size([1, 896, 14, 14]))
x217 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x209, x217)
end = time.time()
print(end-start)
