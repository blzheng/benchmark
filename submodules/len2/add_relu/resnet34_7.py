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
        self.relu15 = ReLU(inplace=True)

    def forward(self, x60, x62):
        x63=operator.add(x60, x62)
        x64=self.relu15(x63)
        return x64

m = M().eval()
x60 = torch.randn(torch.Size([1, 256, 14, 14]))
x62 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x60, x62)
end = time.time()
print(end-start)
