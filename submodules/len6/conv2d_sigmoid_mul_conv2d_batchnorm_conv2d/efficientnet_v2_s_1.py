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
        self.conv2d52 = Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d53 = Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x168, x165):
        x169=self.conv2d52(x168)
        x170=self.sigmoid6(x169)
        x171=operator.mul(x170, x165)
        x172=self.conv2d53(x171)
        x173=self.batchnorm2d39(x172)
        x174=self.conv2d54(x173)
        return x174

m = M().eval()
x168 = torch.randn(torch.Size([1, 32, 1, 1]))
x165 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x168, x165)
end = time.time()
print(end-start)
