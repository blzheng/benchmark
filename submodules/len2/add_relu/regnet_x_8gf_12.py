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
        self.relu39 = ReLU(inplace=True)

    def forward(self, x129, x137):
        x138=operator.add(x129, x137)
        x139=self.relu39(x138)
        return x139

m = M().eval()
x129 = torch.randn(torch.Size([1, 720, 14, 14]))
x137 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x129, x137)
end = time.time()
print(end-start)
