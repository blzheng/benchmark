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
        self.sigmoid0 = Sigmoid()

    def forward(self, x15):
        x16=self.sigmoid0(x15)
        return x16

m = M().eval()
x15 = torch.randn(torch.Size([1, 72, 1, 1]))
start = time.time()
output = m(x15)
end = time.time()
print(end-start)
