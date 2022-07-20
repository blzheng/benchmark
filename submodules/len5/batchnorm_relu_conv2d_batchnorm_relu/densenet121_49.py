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
        self.batchnorm2d102 = BatchNorm2d(736, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu102 = ReLU(inplace=True)
        self.conv2d102 = Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d103 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu103 = ReLU(inplace=True)

    def forward(self, x363):
        x364=self.batchnorm2d102(x363)
        x365=self.relu102(x364)
        x366=self.conv2d102(x365)
        x367=self.batchnorm2d103(x366)
        x368=self.relu103(x367)
        return x368

m = M().eval()
x363 = torch.randn(torch.Size([1, 736, 7, 7]))
start = time.time()
output = m(x363)
end = time.time()
print(end-start)
