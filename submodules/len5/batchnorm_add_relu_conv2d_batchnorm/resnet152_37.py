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
        self.batchnorm2d108 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu103 = ReLU(inplace=True)
        self.conv2d109 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d109 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x357, x350):
        x358=self.batchnorm2d108(x357)
        x359=operator.add(x358, x350)
        x360=self.relu103(x359)
        x361=self.conv2d109(x360)
        x362=self.batchnorm2d109(x361)
        return x362

m = M().eval()
x357 = torch.randn(torch.Size([1, 1024, 14, 14]))
x350 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x357, x350)
end = time.time()
print(end-start)
