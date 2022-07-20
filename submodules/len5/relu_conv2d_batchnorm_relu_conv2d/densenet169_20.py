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
        self.relu43 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x156):
        x157=self.relu43(x156)
        x158=self.conv2d43(x157)
        x159=self.batchnorm2d44(x158)
        x160=self.relu44(x159)
        x161=self.conv2d44(x160)
        return x161

m = M().eval()
x156 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x156)
end = time.time()
print(end-start)
