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
        self.maxpool2d0 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d2 = Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x6):
        x7=self.maxpool2d0(x6)
        x8=self.conv2d2(x7)
        x9=self.batchnorm2d2(x8)
        return x9

m = M().eval()
x6 = torch.randn(torch.Size([1, 64, 224, 224]))
start = time.time()
output = m(x6)
end = time.time()
print(end-start)
