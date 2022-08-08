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
        self.conv2d53 = Conv2d(408, 912, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(912, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)
        self.conv2d54 = Conv2d(912, 912, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=38, bias=False)
        self.batchnorm2d54 = BatchNorm2d(912, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu50 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(912, 912, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(912, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x169):
        x172=self.conv2d53(x169)
        x173=self.batchnorm2d53(x172)
        x174=self.relu49(x173)
        x175=self.conv2d54(x174)
        x176=self.batchnorm2d54(x175)
        x177=self.relu50(x176)
        x178=self.conv2d55(x177)
        x179=self.batchnorm2d55(x178)
        return x179

m = M().eval()
x169 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x169)
end = time.time()
print(end-start)
