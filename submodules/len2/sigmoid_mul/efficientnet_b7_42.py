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
        self.sigmoid42 = Sigmoid()

    def forward(self, x661, x657):
        x662=self.sigmoid42(x661)
        x663=operator.mul(x662, x657)
        return x663

m = M().eval()
x661 = torch.randn(torch.Size([1, 2304, 1, 1]))
x657 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x661, x657)
end = time.time()
print(end-start)
