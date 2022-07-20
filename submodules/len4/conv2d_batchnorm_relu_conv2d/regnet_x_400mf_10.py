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
        self.conv2d17 = Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=10, bias=False)
        self.batchnorm2d17 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x52):
        x53=self.conv2d17(x52)
        x54=self.batchnorm2d17(x53)
        x55=self.relu14(x54)
        x56=self.conv2d18(x55)
        return x56

m = M().eval()
x52 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x52)
end = time.time()
print(end-start)
