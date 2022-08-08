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
        self.batchnorm2d38 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x126):
        x127=self.batchnorm2d38(x126)
        x128=self.relu34(x127)
        x129=self.conv2d39(x128)
        x130=self.batchnorm2d39(x129)
        return x130

m = M().eval()
x126 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x126)
end = time.time()
print(end-start)
