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
        self.conv2d181 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d107 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x540, x536):
        x541=x540.sigmoid()
        x542=operator.mul(x536, x541)
        x543=self.conv2d181(x542)
        x544=self.batchnorm2d107(x543)
        return x544

m = M().eval()
x540 = torch.randn(torch.Size([1, 1344, 1, 1]))
x536 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x540, x536)
end = time.time()
print(end-start)
