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
        self.conv2d16 = Conv2d(12, 120, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()

    def forward(self, x48, x45):
        x49=self.conv2d16(x48)
        x50=self.sigmoid2(x49)
        x51=operator.mul(x50, x45)
        return x51

m = M().eval()
x48 = torch.randn(torch.Size([1, 12, 1, 1]))
x45 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x48, x45)
end = time.time()
print(end-start)
