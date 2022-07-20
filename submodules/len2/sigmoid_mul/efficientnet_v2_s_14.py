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
        self.sigmoid14 = Sigmoid()

    def forward(self, x295, x291):
        x296=self.sigmoid14(x295)
        x297=operator.mul(x296, x291)
        return x297

m = M().eval()
x295 = torch.randn(torch.Size([1, 960, 1, 1]))
x291 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x295, x291)
end = time.time()
print(end-start)
