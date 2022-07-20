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
        self.batchnorm2d46 = BatchNorm2d(336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
        self.batchnorm2d47 = BatchNorm2d(336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x150):
        x151=self.batchnorm2d46(x150)
        x152=self.relu31(x151)
        x153=self.conv2d47(x152)
        x154=self.batchnorm2d47(x153)
        return x154

m = M().eval()
x150 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x150)
end = time.time()
print(end-start)
