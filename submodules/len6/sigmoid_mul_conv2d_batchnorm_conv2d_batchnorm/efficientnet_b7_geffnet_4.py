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
        self.conv2d191 = Conv2d(1344, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d113 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d192 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d114 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x570, x566):
        x571=x570.sigmoid()
        x572=operator.mul(x566, x571)
        x573=self.conv2d191(x572)
        x574=self.batchnorm2d113(x573)
        x575=self.conv2d192(x574)
        x576=self.batchnorm2d114(x575)
        return x576

m = M().eval()
x570 = torch.randn(torch.Size([1, 1344, 1, 1]))
x566 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x570, x566)
end = time.time()
print(end-start)
