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
        self.batchnorm2d17 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x59, x62):
        x60=self.batchnorm2d17(x59)
        x63=operator.add(x60, x62)
        x64=self.relu15(x63)
        x65=self.conv2d19(x64)
        return x65

m = M().eval()
x59 = torch.randn(torch.Size([1, 256, 14, 14]))
x62 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x59, x62)
end = time.time()
print(end-start)
