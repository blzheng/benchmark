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
        self.conv2d146 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x456, x451):
        x457=operator.mul(x456, x451)
        x458=self.conv2d146(x457)
        return x458

m = M().eval()
x456 = torch.randn(torch.Size([1, 1344, 1, 1]))
x451 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x456, x451)
end = time.time()
print(end-start)
