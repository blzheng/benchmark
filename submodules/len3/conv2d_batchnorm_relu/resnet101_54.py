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
        self.conv2d83 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d83 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu79 = ReLU(inplace=True)

    def forward(self, x273):
        x274=self.conv2d83(x273)
        x275=self.batchnorm2d83(x274)
        x276=self.relu79(x275)
        return x276

m = M().eval()
x273 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x273)
end = time.time()
print(end-start)
