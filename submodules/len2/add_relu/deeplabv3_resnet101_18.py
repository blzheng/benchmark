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
        self.relu55 = ReLU(inplace=True)

    def forward(self, x200, x192):
        x201=operator.add(x200, x192)
        x202=self.relu55(x201)
        return x202

m = M().eval()
x200 = torch.randn(torch.Size([1, 1024, 28, 28]))
x192 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x200, x192)
end = time.time()
print(end-start)
