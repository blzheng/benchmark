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
        self.maxpool2d3 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d10 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d10 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x33):
        x34=self.maxpool2d3(x33)
        x35=self.conv2d10(x34)
        x36=self.batchnorm2d10(x35)
        return x36

m = M().eval()
x33 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x33)
end = time.time()
print(end-start)