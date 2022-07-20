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
        self.conv2d298 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d192 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x958, x943):
        x959=operator.add(x958, x943)
        x960=self.conv2d298(x959)
        x961=self.batchnorm2d192(x960)
        return x961

m = M().eval()
x958 = torch.randn(torch.Size([1, 384, 7, 7]))
x943 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x958, x943)
end = time.time()
print(end-start)
