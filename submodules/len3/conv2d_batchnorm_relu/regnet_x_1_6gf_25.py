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
        self.conv2d40 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)

    def forward(self, x129):
        x130=self.conv2d40(x129)
        x131=self.batchnorm2d40(x130)
        x132=self.relu37(x131)
        return x132

m = M().eval()
x129 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x129)
end = time.time()
print(end-start)
