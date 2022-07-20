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
        self.conv2d31 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x89, x94):
        x95=operator.mul(x89, x94)
        x96=self.conv2d31(x95)
        x97=self.batchnorm2d17(x96)
        return x97

m = M().eval()
x89 = torch.randn(torch.Size([1, 288, 56, 56]))
x94 = torch.randn(torch.Size([1, 288, 1, 1]))
start = time.time()
output = m(x89, x94)
end = time.time()
print(end-start)
