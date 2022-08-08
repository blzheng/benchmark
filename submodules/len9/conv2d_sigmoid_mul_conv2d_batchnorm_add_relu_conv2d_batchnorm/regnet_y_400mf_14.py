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
        self.conv2d78 = Conv2d(110, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d79 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU(inplace=True)
        self.conv2d80 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x244, x241, x235):
        x245=self.conv2d78(x244)
        x246=self.sigmoid14(x245)
        x247=operator.mul(x246, x241)
        x248=self.conv2d79(x247)
        x249=self.batchnorm2d49(x248)
        x250=operator.add(x235, x249)
        x251=self.relu60(x250)
        x252=self.conv2d80(x251)
        x253=self.batchnorm2d50(x252)
        return x253

m = M().eval()
x244 = torch.randn(torch.Size([1, 110, 1, 1]))
x241 = torch.randn(torch.Size([1, 440, 7, 7]))
x235 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x244, x241, x235)
end = time.time()
print(end-start)
