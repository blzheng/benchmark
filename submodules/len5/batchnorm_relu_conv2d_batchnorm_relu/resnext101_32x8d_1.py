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
        self.batchnorm2d5 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d6 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d6 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x17):
        x18=self.batchnorm2d5(x17)
        x19=self.relu4(x18)
        x20=self.conv2d6(x19)
        x21=self.batchnorm2d6(x20)
        x22=self.relu4(x21)
        return x22

m = M().eval()
x17 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x17)
end = time.time()
print(end-start)
