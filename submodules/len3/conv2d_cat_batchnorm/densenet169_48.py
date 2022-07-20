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
        self.conv2d107 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d110 = BatchNorm2d(736, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x382, x369, x376, x390):
        x383=self.conv2d107(x382)
        x391=torch.cat([x369, x376, x383, x390], 1)
        x392=self.batchnorm2d110(x391)
        return x392

m = M().eval()
x382 = torch.randn(torch.Size([1, 128, 7, 7]))
x369 = torch.randn(torch.Size([1, 640, 7, 7]))
x376 = torch.randn(torch.Size([1, 32, 7, 7]))
x390 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x382, x369, x376, x390)
end = time.time()
print(end-start)
