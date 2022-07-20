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
        self.conv2d177 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d119 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x572, x567):
        x573=operator.mul(x572, x567)
        x574=self.conv2d177(x573)
        x575=self.batchnorm2d119(x574)
        return x575

m = M().eval()
x572 = torch.randn(torch.Size([1, 1344, 1, 1]))
x567 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x572, x567)
end = time.time()
print(end-start)
