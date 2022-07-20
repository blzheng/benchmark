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
        self.sigmoid51 = Sigmoid()
        self.conv2d292 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d188 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x937, x933):
        x938=self.sigmoid51(x937)
        x939=operator.mul(x938, x933)
        x940=self.conv2d292(x939)
        x941=self.batchnorm2d188(x940)
        return x941

m = M().eval()
x937 = torch.randn(torch.Size([1, 2304, 1, 1]))
x933 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x937, x933)
end = time.time()
print(end-start)
