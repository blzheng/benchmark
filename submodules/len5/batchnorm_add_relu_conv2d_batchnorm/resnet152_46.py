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
        self.batchnorm2d135 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu130 = ReLU(inplace=True)
        self.conv2d136 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d136 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x447, x440):
        x448=self.batchnorm2d135(x447)
        x449=operator.add(x448, x440)
        x450=self.relu130(x449)
        x451=self.conv2d136(x450)
        x452=self.batchnorm2d136(x451)
        return x452

m = M().eval()
x447 = torch.randn(torch.Size([1, 1024, 14, 14]))
x440 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x447, x440)
end = time.time()
print(end-start)
