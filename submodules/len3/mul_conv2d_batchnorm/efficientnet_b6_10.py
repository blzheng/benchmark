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
        self.conv2d52 = Conv2d(432, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x159, x154):
        x160=operator.mul(x159, x154)
        x161=self.conv2d52(x160)
        x162=self.batchnorm2d30(x161)
        return x162

m = M().eval()
x159 = torch.randn(torch.Size([1, 432, 1, 1]))
x154 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x159, x154)
end = time.time()
print(end-start)
