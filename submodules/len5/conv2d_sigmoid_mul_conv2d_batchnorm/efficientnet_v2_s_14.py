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
        self.conv2d92 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d93 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x294, x291):
        x295=self.conv2d92(x294)
        x296=self.sigmoid14(x295)
        x297=operator.mul(x296, x291)
        x298=self.conv2d93(x297)
        x299=self.batchnorm2d63(x298)
        return x299

m = M().eval()
x294 = torch.randn(torch.Size([1, 40, 1, 1]))
x291 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x294, x291)
end = time.time()
print(end-start)
