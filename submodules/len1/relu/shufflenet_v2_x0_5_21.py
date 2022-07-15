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
        self.relu21 = ReLU(inplace=True)

    def forward(self, x211):
        x212=self.relu21(x211)
        return x212

m = M().eval()
x211 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x211)
end = time.time()
print(end-start)
