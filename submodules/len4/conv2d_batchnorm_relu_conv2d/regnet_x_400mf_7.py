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
        self.conv2d13 = Conv2d(64, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(160, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=10, bias=False)

    def forward(self, x37):
        x40=self.conv2d13(x37)
        x41=self.batchnorm2d13(x40)
        x42=self.relu10(x41)
        x43=self.conv2d14(x42)
        return x43

m = M().eval()
x37 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
