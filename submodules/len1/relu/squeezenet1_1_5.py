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
        self.relu5 = ReLU(inplace=True)

    def forward(self, x13):
        x14=self.relu5(x13)
        return x14

m = M().eval()
x13 = torch.randn(torch.Size([1, 64, 55, 55]))
start = time.time()
output = m(x13)
end = time.time()
print(end-start)
