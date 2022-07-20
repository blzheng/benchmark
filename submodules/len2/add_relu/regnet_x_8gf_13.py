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
        self.relu42 = ReLU(inplace=True)

    def forward(self, x139, x147):
        x148=operator.add(x139, x147)
        x149=self.relu42(x148)
        return x149

m = M().eval()
x139 = torch.randn(torch.Size([1, 720, 14, 14]))
x147 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x139, x147)
end = time.time()
print(end-start)