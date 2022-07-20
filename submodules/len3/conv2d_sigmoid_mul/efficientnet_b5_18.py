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
        self.conv2d91 = Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()

    def forward(self, x283, x280):
        x284=self.conv2d91(x283)
        x285=self.sigmoid18(x284)
        x286=operator.mul(x285, x280)
        return x286

m = M().eval()
x283 = torch.randn(torch.Size([1, 32, 1, 1]))
x280 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x283, x280)
end = time.time()
print(end-start)
