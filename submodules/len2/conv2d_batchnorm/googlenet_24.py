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
        self.conv2d24 = Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x85):
        x95=self.conv2d24(x85)
        x96=self.batchnorm2d24(x95)
        return x96

m = M().eval()
x85 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x85)
end = time.time()
print(end-start)
