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
        self.conv2d90 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()

    def forward(self, x282, x279):
        x283=self.conv2d90(x282)
        x284=self.sigmoid18(x283)
        x285=operator.mul(x284, x279)
        return x285

m = M().eval()
x282 = torch.randn(torch.Size([1, 20, 1, 1]))
x279 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x282, x279)
end = time.time()
print(end-start)
