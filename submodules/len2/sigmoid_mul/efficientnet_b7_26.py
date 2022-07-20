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
        self.sigmoid26 = Sigmoid()

    def forward(self, x409, x405):
        x410=self.sigmoid26(x409)
        x411=operator.mul(x410, x405)
        return x411

m = M().eval()
x409 = torch.randn(torch.Size([1, 960, 1, 1]))
x405 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x409, x405)
end = time.time()
print(end-start)
