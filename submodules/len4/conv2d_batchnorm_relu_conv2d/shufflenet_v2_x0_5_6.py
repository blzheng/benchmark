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
        self.conv2d23 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)

    def forward(self, x143):
        x144=self.conv2d23(x143)
        x145=self.batchnorm2d23(x144)
        x146=self.relu15(x145)
        x147=self.conv2d24(x146)
        return x147

m = M().eval()
x143 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x143)
end = time.time()
print(end-start)
