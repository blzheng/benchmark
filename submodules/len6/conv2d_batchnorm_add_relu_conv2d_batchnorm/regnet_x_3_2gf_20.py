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
        self.conv2d57 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu54 = ReLU(inplace=True)
        self.conv2d58 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x185, x179):
        x186=self.conv2d57(x185)
        x187=self.batchnorm2d57(x186)
        x188=operator.add(x179, x187)
        x189=self.relu54(x188)
        x190=self.conv2d58(x189)
        x191=self.batchnorm2d58(x190)
        return x191

m = M().eval()
x185 = torch.randn(torch.Size([1, 432, 14, 14]))
x179 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x185, x179)
end = time.time()
print(end-start)
