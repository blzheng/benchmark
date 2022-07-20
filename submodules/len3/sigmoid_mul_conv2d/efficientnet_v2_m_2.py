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
        self.sigmoid2 = Sigmoid()
        self.conv2d38 = Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x124, x120):
        x125=self.sigmoid2(x124)
        x126=operator.mul(x125, x120)
        x127=self.conv2d38(x126)
        return x127

m = M().eval()
x124 = torch.randn(torch.Size([1, 640, 1, 1]))
x120 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x124, x120)
end = time.time()
print(end-start)
