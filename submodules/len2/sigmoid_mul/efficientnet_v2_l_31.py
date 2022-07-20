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
        self.sigmoid31 = Sigmoid()

    def forward(self, x617, x613):
        x618=self.sigmoid31(x617)
        x619=operator.mul(x618, x613)
        return x619

m = M().eval()
x617 = torch.randn(torch.Size([1, 2304, 1, 1]))
x613 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x617, x613)
end = time.time()
print(end-start)
