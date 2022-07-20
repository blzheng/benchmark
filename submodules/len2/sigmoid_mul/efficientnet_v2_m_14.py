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

    def forward(self, x314, x310):
        x315=self.sigmoid14(x314)
        x316=operator.mul(x315, x310)
        return x316

m = M().eval()
x314 = torch.randn(torch.Size([1, 1056, 1, 1]))
x310 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x314, x310)
end = time.time()
print(end-start)
