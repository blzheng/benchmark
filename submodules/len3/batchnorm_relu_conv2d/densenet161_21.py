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
        self.batchnorm2d22 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d22 = Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x80):
        x81=self.batchnorm2d22(x80)
        x82=self.relu22(x81)
        x83=self.conv2d22(x82)
        return x83

m = M().eval()
x80 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x80)
end = time.time()
print(end-start)
