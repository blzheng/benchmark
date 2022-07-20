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
        self.sigmoid23 = Sigmoid()

    def forward(self, x364, x360):
        x365=self.sigmoid23(x364)
        x366=operator.mul(x365, x360)
        return x366

m = M().eval()
x364 = torch.randn(torch.Size([1, 864, 1, 1]))
x360 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x364, x360)
end = time.time()
print(end-start)
