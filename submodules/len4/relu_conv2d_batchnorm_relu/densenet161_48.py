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
        self.relu99 = ReLU(inplace=True)
        self.conv2d99 = Conv2d(1824, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d100 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu100 = ReLU(inplace=True)

    def forward(self, x352):
        x353=self.relu99(x352)
        x354=self.conv2d99(x353)
        x355=self.batchnorm2d100(x354)
        x356=self.relu100(x355)
        return x356

m = M().eval()
x352 = torch.randn(torch.Size([1, 1824, 14, 14]))
start = time.time()
output = m(x352)
end = time.time()
print(end-start)
