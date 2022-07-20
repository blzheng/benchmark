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
        self.conv2d45 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu630 = ReLU6(inplace=True)
        self.conv2d46 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)

    def forward(self, x129):
        x130=self.conv2d45(x129)
        x131=self.batchnorm2d45(x130)
        x132=self.relu630(x131)
        x133=self.conv2d46(x132)
        return x133

m = M().eval()
x129 = torch.randn(torch.Size([1, 160, 7, 7]))
start = time.time()
output = m(x129)
end = time.time()
print(end-start)
