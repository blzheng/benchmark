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
        self.batchnorm2d35 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d38 = Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x130):
        x131=self.batchnorm2d35(x130)
        x132=self.conv2d38(x131)
        x133=self.batchnorm2d36(x132)
        return x133

m = M().eval()
x130 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x130)
end = time.time()
print(end-start)
