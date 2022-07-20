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
        self.sigmoid27 = Sigmoid()

    def forward(self, x425):
        x426=self.sigmoid27(x425)
        return x426

m = M().eval()
x425 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x425)
end = time.time()
print(end-start)