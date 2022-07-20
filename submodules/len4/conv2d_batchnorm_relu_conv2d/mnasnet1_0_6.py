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
        self.conv2d9 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)

    def forward(self, x25):
        x26=self.conv2d9(x25)
        x27=self.batchnorm2d9(x26)
        x28=self.relu6(x27)
        x29=self.conv2d10(x28)
        return x29

m = M().eval()
x25 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x25)
end = time.time()
print(end-start)
