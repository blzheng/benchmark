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
        self.conv2d21 = Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)
        self.batchnorm2d21 = BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d22 = Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x124):
        x125=self.conv2d21(x124)
        x126=self.batchnorm2d21(x125)
        x127=self.conv2d22(x126)
        x128=self.batchnorm2d22(x127)
        return x128

m = M().eval()
x124 = torch.randn(torch.Size([1, 116, 14, 14]))
start = time.time()
output = m(x124)
end = time.time()
print(end-start)
