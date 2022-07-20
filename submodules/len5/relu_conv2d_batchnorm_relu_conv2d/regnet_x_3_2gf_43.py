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
        self.relu70 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(1008, 1008, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=21, bias=False)
        self.batchnorm2d75 = BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu71 = ReLU(inplace=True)
        self.conv2d76 = Conv2d(1008, 1008, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x243):
        x244=self.relu70(x243)
        x245=self.conv2d75(x244)
        x246=self.batchnorm2d75(x245)
        x247=self.relu71(x246)
        x248=self.conv2d76(x247)
        return x248

m = M().eval()
x243 = torch.randn(torch.Size([1, 1008, 14, 14]))
start = time.time()
output = m(x243)
end = time.time()
print(end-start)
