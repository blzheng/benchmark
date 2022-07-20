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
        self.maxpool2d7 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d32 = Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x105):
        x121=self.maxpool2d7(x105)
        x122=self.conv2d32(x121)
        return x122

m = M().eval()
x105 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x105)
end = time.time()
print(end-start)
