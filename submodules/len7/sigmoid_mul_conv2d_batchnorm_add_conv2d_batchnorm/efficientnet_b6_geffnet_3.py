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
        self.conv2d152 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d90 = BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d153 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d91 = BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x452, x448, x442):
        x453=x452.sigmoid()
        x454=operator.mul(x448, x453)
        x455=self.conv2d152(x454)
        x456=self.batchnorm2d90(x455)
        x457=operator.add(x456, x442)
        x458=self.conv2d153(x457)
        x459=self.batchnorm2d91(x458)
        return x459

m = M().eval()
x452 = torch.randn(torch.Size([1, 1200, 1, 1]))
x448 = torch.randn(torch.Size([1, 1200, 14, 14]))
x442 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x452, x448, x442)
end = time.time()
print(end-start)
