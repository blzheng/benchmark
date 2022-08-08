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
        self.conv2d82 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d82 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu79 = ReLU(inplace=True)

    def forward(self, x271):
        x272=self.relu76(x271)
        x273=self.conv2d82(x272)
        x274=self.batchnorm2d82(x273)
        x275=self.relu79(x274)
        return x275

m = M().eval()
x271 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x271)
end = time.time()
print(end-start)
