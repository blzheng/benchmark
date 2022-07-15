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
        self.sigmoid56 = Sigmoid()

    def forward(self, x1015):
        x1016=self.sigmoid56(x1015)
        return x1016

m = M().eval()
x1015 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x1015)
end = time.time()
print(end-start)
