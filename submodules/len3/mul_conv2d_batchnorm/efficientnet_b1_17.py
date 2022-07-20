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
        self.conv2d88 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x268, x263):
        x269=operator.mul(x268, x263)
        x270=self.conv2d88(x269)
        x271=self.batchnorm2d52(x270)
        return x271

m = M().eval()
x268 = torch.randn(torch.Size([1, 1152, 1, 1]))
x263 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x268, x263)
end = time.time()
print(end-start)
