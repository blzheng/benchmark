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
        self.sigmoid34 = Sigmoid()
        self.conv2d172 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d102 = BatchNorm2d(344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x536, x532):
        x537=self.sigmoid34(x536)
        x538=operator.mul(x537, x532)
        x539=self.conv2d172(x538)
        x540=self.batchnorm2d102(x539)
        return x540

m = M().eval()
x536 = torch.randn(torch.Size([1, 2064, 1, 1]))
x532 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x536, x532)
end = time.time()
print(end-start)
