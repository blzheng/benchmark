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
        self.relu112 = ReLU(inplace=True)
        self.conv2d116 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d116 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d117 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x382):
        x383=self.relu112(x382)
        x384=self.conv2d116(x383)
        x385=self.batchnorm2d116(x384)
        x386=self.relu112(x385)
        x387=self.conv2d117(x386)
        return x387

m = M().eval()
x382 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x382)
end = time.time()
print(end-start)
