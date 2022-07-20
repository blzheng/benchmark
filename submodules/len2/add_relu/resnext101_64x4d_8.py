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
        self.relu25 = ReLU(inplace=True)

    def forward(self, x98, x90):
        x99=operator.add(x98, x90)
        x100=self.relu25(x99)
        return x100

m = M().eval()
x98 = torch.randn(torch.Size([1, 1024, 14, 14]))
x90 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x98, x90)
end = time.time()
print(end-start)
