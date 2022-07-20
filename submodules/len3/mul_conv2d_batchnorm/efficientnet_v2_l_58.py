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
        self.conv2d327 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d209 = BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x1048, x1043):
        x1049=operator.mul(x1048, x1043)
        x1050=self.conv2d327(x1049)
        x1051=self.batchnorm2d209(x1050)
        return x1051

m = M().eval()
x1048 = torch.randn(torch.Size([1, 3840, 1, 1]))
x1043 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1048, x1043)
end = time.time()
print(end-start)
