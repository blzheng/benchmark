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
        self.conv2d141 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()
        self.conv2d142 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d98 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x458, x455):
        x459=self.conv2d141(x458)
        x460=self.sigmoid21(x459)
        x461=operator.mul(x460, x455)
        x462=self.conv2d142(x461)
        x463=self.batchnorm2d98(x462)
        return x463

m = M().eval()
x458 = torch.randn(torch.Size([1, 56, 1, 1]))
x455 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x458, x455)
end = time.time()
print(end-start)
