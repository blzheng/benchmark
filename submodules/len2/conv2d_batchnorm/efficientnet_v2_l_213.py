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
        self.conv2d333 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d213 = BatchNorm2d(3840, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x1069):
        x1070=self.conv2d333(x1069)
        x1071=self.batchnorm2d213(x1070)
        return x1071

m = M().eval()
x1069 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x1069)
end = time.time()
print(end-start)
