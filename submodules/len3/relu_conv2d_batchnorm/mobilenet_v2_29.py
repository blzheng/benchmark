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
        self.relu629 = ReLU6(inplace=True)
        self.conv2d44 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x125):
        x126=self.relu629(x125)
        x127=self.conv2d44(x126)
        x128=self.batchnorm2d44(x127)
        return x128

m = M().eval()
x125 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x125)
end = time.time()
print(end-start)
