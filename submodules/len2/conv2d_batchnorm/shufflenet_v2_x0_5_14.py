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
        self.conv2d14 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x80):
        x81=self.conv2d14(x80)
        x82=self.batchnorm2d14(x81)
        return x82

m = M().eval()
x80 = torch.randn(torch.Size([1, 24, 28, 28]))
start = time.time()
output = m(x80)
end = time.time()
print(end-start)