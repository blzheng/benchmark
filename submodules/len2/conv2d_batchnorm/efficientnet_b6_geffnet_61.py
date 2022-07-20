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
        self.conv2d103 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x308):
        x309=self.conv2d103(x308)
        x310=self.batchnorm2d61(x309)
        return x310

m = M().eval()
x308 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x308)
end = time.time()
print(end-start)
