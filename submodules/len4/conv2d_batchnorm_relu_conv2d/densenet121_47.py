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
        self.conv2d98 = Conv2d(672, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d99 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu99 = ReLU(inplace=True)
        self.conv2d99 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x351):
        x352=self.conv2d98(x351)
        x353=self.batchnorm2d99(x352)
        x354=self.relu99(x353)
        x355=self.conv2d99(x354)
        return x355

m = M().eval()
x351 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x351)
end = time.time()
print(end-start)
