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
        self.conv2d83 = Conv2d(1440, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d84 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x297):
        x298=self.conv2d83(x297)
        x299=self.batchnorm2d84(x298)
        return x299

m = M().eval()
x297 = torch.randn(torch.Size([1, 1440, 14, 14]))
start = time.time()
output = m(x297)
end = time.time()
print(end-start)
