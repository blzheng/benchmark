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
        self.relu105 = ReLU(inplace=True)
        self.conv2d105 = Conv2d(1968, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d106 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu106 = ReLU(inplace=True)

    def forward(self, x373):
        x374=self.relu105(x373)
        x375=self.conv2d105(x374)
        x376=self.batchnorm2d106(x375)
        x377=self.relu106(x376)
        return x377

m = M().eval()
x373 = torch.randn(torch.Size([1, 1968, 14, 14]))
start = time.time()
output = m(x373)
end = time.time()
print(end-start)
