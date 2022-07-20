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
        self.conv2d171 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid27 = Sigmoid()

    def forward(self, x554, x551):
        x555=self.conv2d171(x554)
        x556=self.sigmoid27(x555)
        x557=operator.mul(x556, x551)
        return x557

m = M().eval()
x554 = torch.randn(torch.Size([1, 56, 1, 1]))
x551 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x554, x551)
end = time.time()
print(end-start)
