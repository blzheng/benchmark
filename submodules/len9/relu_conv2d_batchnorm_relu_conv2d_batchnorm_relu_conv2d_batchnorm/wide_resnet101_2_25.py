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
        self.relu76 = ReLU(inplace=True)
        self.conv2d82 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d82 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu79 = ReLU(inplace=True)
        self.conv2d83 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d83 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d84 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d84 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x269):
        x270=self.relu76(x269)
        x271=self.conv2d82(x270)
        x272=self.batchnorm2d82(x271)
        x273=self.relu79(x272)
        x274=self.conv2d83(x273)
        x275=self.batchnorm2d83(x274)
        x276=self.relu79(x275)
        x277=self.conv2d84(x276)
        x278=self.batchnorm2d84(x277)
        return x278

m = M().eval()
x269 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x269)
end = time.time()
print(end-start)
