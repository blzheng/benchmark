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
        self.conv2d45 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(864, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x128, x120):
        x129=operator.add(x128, x120)
        x130=self.conv2d45(x129)
        x131=self.batchnorm2d45(x130)
        return x131

m = M().eval()
x128 = torch.randn(torch.Size([1, 144, 7, 7]))
x120 = torch.randn(torch.Size([1, 144, 7, 7]))
start = time.time()
output = m(x128, x120)
end = time.time()
print(end-start)
