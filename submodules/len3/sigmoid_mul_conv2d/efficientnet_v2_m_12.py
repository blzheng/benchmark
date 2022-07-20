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
        self.sigmoid12 = Sigmoid()
        self.conv2d88 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x282, x278):
        x283=self.sigmoid12(x282)
        x284=operator.mul(x283, x278)
        x285=self.conv2d88(x284)
        return x285

m = M().eval()
x282 = torch.randn(torch.Size([1, 1056, 1, 1]))
x278 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x282, x278)
end = time.time()
print(end-start)
