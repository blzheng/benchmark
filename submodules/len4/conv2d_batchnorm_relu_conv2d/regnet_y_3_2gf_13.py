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
        self.conv2d64 = Conv2d(576, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)
        self.conv2d65 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)

    def forward(self, x201):
        x202=self.conv2d64(x201)
        x203=self.batchnorm2d40(x202)
        x204=self.relu49(x203)
        x205=self.conv2d65(x204)
        return x205

m = M().eval()
x201 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x201)
end = time.time()
print(end-start)
