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
        self.conv2d128 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d129 = Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x377, x373, x367):
        x378=x377.sigmoid()
        x379=operator.mul(x373, x378)
        x380=self.conv2d128(x379)
        x381=self.batchnorm2d76(x380)
        x382=operator.add(x381, x367)
        x383=self.conv2d129(x382)
        return x383

m = M().eval()
x377 = torch.randn(torch.Size([1, 2304, 1, 1]))
x373 = torch.randn(torch.Size([1, 2304, 7, 7]))
x367 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x377, x373, x367)
end = time.time()
print(end-start)
