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
        self.sigmoid28 = Sigmoid()
        self.conv2d142 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d84 = BatchNorm2d(304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x440, x436):
        x441=self.sigmoid28(x440)
        x442=operator.mul(x441, x436)
        x443=self.conv2d142(x442)
        x444=self.batchnorm2d84(x443)
        return x444

m = M().eval()
x440 = torch.randn(torch.Size([1, 1824, 1, 1]))
x436 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x440, x436)
end = time.time()
print(end-start)
