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
        self.conv2d101 = Conv2d(336, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu79 = ReLU()
        self.conv2d102 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()
        self.conv2d103 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x320, x319):
        x321=self.conv2d101(x320)
        x322=self.relu79(x321)
        x323=self.conv2d102(x322)
        x324=self.sigmoid19(x323)
        x325=operator.mul(x324, x319)
        x326=self.conv2d103(x325)
        x327=self.batchnorm2d63(x326)
        return x327

m = M().eval()
x320 = torch.randn(torch.Size([1, 336, 1, 1]))
x319 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x320, x319)
end = time.time()
print(end-start)
