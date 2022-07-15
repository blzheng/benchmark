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
        self.sigmoid52 = Sigmoid()

    def forward(self, x819):
        x820=self.sigmoid52(x819)
        return x820

m = M().eval()
x819 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x819)
end = time.time()
print(end-start)
