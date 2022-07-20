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
        self.conv2d46 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)

    def forward(self, x150):
        x151=self.conv2d46(x150)
        x152=self.batchnorm2d46(x151)
        x153=self.relu43(x152)
        return x153

m = M().eval()
x150 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x150)
end = time.time()
print(end-start)
