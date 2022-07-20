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
        self.sigmoid21 = Sigmoid()

    def forward(self, x330, x326):
        x331=self.sigmoid21(x330)
        x332=operator.mul(x331, x326)
        return x332

m = M().eval()
x330 = torch.randn(torch.Size([1, 1056, 1, 1]))
x326 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x330, x326)
end = time.time()
print(end-start)
