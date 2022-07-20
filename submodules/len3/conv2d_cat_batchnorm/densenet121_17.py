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
        self.conv2d42 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x153, x140, x147, x161):
        x154=self.conv2d42(x153)
        x162=torch.cat([x140, x147, x154, x161], 1)
        x163=self.batchnorm2d45(x162)
        return x163

m = M().eval()
x153 = torch.randn(torch.Size([1, 128, 14, 14]))
x140 = torch.randn(torch.Size([1, 256, 14, 14]))
x147 = torch.randn(torch.Size([1, 32, 14, 14]))
x161 = torch.randn(torch.Size([1, 32, 14, 14]))
start = time.time()
output = m(x153, x140, x147, x161)
end = time.time()
print(end-start)
