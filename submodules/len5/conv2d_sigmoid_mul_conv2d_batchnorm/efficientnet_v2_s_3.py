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
        self.conv2d37 = Conv2d(32, 512, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = Sigmoid()
        self.conv2d38 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x120, x117):
        x121=self.conv2d37(x120)
        x122=self.sigmoid3(x121)
        x123=operator.mul(x122, x117)
        x124=self.conv2d38(x123)
        x125=self.batchnorm2d30(x124)
        return x125

m = M().eval()
x120 = torch.randn(torch.Size([1, 32, 1, 1]))
x117 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x120, x117)
end = time.time()
print(end-start)
