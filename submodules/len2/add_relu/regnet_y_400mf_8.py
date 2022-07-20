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
        self.relu36 = ReLU(inplace=True)

    def forward(self, x137, x151):
        x152=operator.add(x137, x151)
        x153=self.relu36(x152)
        return x153

m = M().eval()
x137 = torch.randn(torch.Size([1, 208, 14, 14]))
x151 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x137, x151)
end = time.time()
print(end-start)
