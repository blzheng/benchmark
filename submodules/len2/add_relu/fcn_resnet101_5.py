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
        self.relu16 = ReLU(inplace=True)

    def forward(self, x68, x60):
        x69=operator.add(x68, x60)
        x70=self.relu16(x69)
        return x70

m = M().eval()
x68 = torch.randn(torch.Size([1, 512, 28, 28]))
x60 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x68, x60)
end = time.time()
print(end-start)
