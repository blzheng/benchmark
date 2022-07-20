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
        self.conv2d63 = Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu42 = ReLU(inplace=True)

    def forward(self, x204):
        x205=self.conv2d63(x204)
        x206=self.batchnorm2d63(x205)
        x207=self.relu42(x206)
        return x207

m = M().eval()
x204 = torch.randn(torch.Size([1, 320, 7, 7]))
start = time.time()
output = m(x204)
end = time.time()
print(end-start)
