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
        self.conv2d97 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d97 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)
        self.conv2d98 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d98 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x320):
        x321=self.conv2d97(x320)
        x322=self.batchnorm2d97(x321)
        x323=self.relu94(x322)
        x324=self.conv2d98(x323)
        x325=self.batchnorm2d98(x324)
        x326=self.relu94(x325)
        return x326

m = M().eval()
x320 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x320)
end = time.time()
print(end-start)
