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
        self.batchnorm2d6 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d7 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x32):
        x33=self.batchnorm2d6(x32)
        x34=self.relu4(x33)
        x35=self.conv2d7(x34)
        x36=self.batchnorm2d7(x35)
        return x36

m = M().eval()
x32 = torch.randn(torch.Size([1, 24, 28, 28]))
start = time.time()
output = m(x32)
end = time.time()
print(end-start)
