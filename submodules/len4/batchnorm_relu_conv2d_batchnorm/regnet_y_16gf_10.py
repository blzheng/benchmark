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
        self.batchnorm2d31 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(1232, 1232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)
        self.batchnorm2d32 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x154):
        x155=self.batchnorm2d31(x154)
        x156=self.relu37(x155)
        x157=self.conv2d50(x156)
        x158=self.batchnorm2d32(x157)
        return x158

m = M().eval()
x154 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x154)
end = time.time()
print(end-start)
