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
        self.batchnorm2d34 = BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d35 = Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(288, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x129):
        x130=self.batchnorm2d34(x129)
        x131=torch.nn.functional.relu(x130,inplace=True)
        x132=self.conv2d35(x131)
        x133=self.batchnorm2d35(x132)
        x134=torch.nn.functional.relu(x133,inplace=True)
        return x134

m = M().eval()
x129 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x129)
end = time.time()
print(end-start)
