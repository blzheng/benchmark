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
        self.relu61 = ReLU(inplace=True)

    def forward(self, x218, x210):
        x219=operator.add(x218, x210)
        x220=self.relu61(x219)
        return x220

m = M().eval()
x218 = torch.randn(torch.Size([1, 1024, 14, 14]))
x210 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x218, x210)
end = time.time()
print(end-start)
