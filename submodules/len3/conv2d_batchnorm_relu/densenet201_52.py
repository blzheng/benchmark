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
        self.conv2d105 = Conv2d(1312, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d106 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu106 = ReLU(inplace=True)

    def forward(self, x374):
        x375=self.conv2d105(x374)
        x376=self.batchnorm2d106(x375)
        x377=self.relu106(x376)
        return x377

m = M().eval()
x374 = torch.randn(torch.Size([1, 1312, 14, 14]))
start = time.time()
output = m(x374)
end = time.time()
print(end-start)
