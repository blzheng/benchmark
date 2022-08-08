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
        self.conv2d81 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d81 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x268, x262):
        x269=self.conv2d81(x268)
        x270=self.batchnorm2d81(x269)
        x271=operator.add(x270, x262)
        return x271

m = M().eval()
x268 = torch.randn(torch.Size([1, 256, 28, 28]))
x262 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x268, x262)
end = time.time()
print(end-start)
