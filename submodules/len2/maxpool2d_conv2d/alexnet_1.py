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
        self.maxpool2d1 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d2 = Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x5):
        x6=self.maxpool2d1(x5)
        x7=self.conv2d2(x6)
        return x7

m = M().eval()
x5 = torch.randn(torch.Size([1, 192, 27, 27]))
start = time.time()
output = m(x5)
end = time.time()
print(end-start)
