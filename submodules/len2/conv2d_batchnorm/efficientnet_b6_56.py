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
        self.conv2d94 = Conv2d(864, 864, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=864, bias=False)
        self.batchnorm2d56 = BatchNorm2d(864, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x293):
        x294=self.conv2d94(x293)
        x295=self.batchnorm2d56(x294)
        return x295

m = M().eval()
x293 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x293)
end = time.time()
print(end-start)
