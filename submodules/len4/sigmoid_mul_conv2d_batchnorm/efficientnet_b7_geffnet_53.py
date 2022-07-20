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
        self.conv2d266 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d158 = BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x793, x789):
        x794=x793.sigmoid()
        x795=operator.mul(x789, x794)
        x796=self.conv2d266(x795)
        x797=self.batchnorm2d158(x796)
        return x797

m = M().eval()
x793 = torch.randn(torch.Size([1, 3840, 1, 1]))
x789 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x793, x789)
end = time.time()
print(end-start)
