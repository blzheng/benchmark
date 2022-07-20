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
        self.batchnorm2d152 = BatchNorm2d(1408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu152 = ReLU(inplace=True)
        self.conv2d152 = Conv2d(1408, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d153 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu153 = ReLU(inplace=True)

    def forward(self, x538):
        x539=self.batchnorm2d152(x538)
        x540=self.relu152(x539)
        x541=self.conv2d152(x540)
        x542=self.batchnorm2d153(x541)
        x543=self.relu153(x542)
        return x543

m = M().eval()
x538 = torch.randn(torch.Size([1, 1408, 7, 7]))
start = time.time()
output = m(x538)
end = time.time()
print(end-start)
