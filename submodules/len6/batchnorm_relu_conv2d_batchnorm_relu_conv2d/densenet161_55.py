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
        self.batchnorm2d114 = BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu114 = ReLU(inplace=True)
        self.conv2d114 = Conv2d(1104, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d115 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu115 = ReLU(inplace=True)
        self.conv2d115 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x405):
        x406=self.batchnorm2d114(x405)
        x407=self.relu114(x406)
        x408=self.conv2d114(x407)
        x409=self.batchnorm2d115(x408)
        x410=self.relu115(x409)
        x411=self.conv2d115(x410)
        return x411

m = M().eval()
x405 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x405)
end = time.time()
print(end-start)
