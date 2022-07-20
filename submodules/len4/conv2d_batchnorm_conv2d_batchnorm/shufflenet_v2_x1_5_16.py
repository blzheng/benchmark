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
        self.conv2d47 = Conv2d(352, 352, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=352, bias=False)
        self.batchnorm2d47 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d48 = Conv2d(352, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x302):
        x303=self.conv2d47(x302)
        x304=self.batchnorm2d47(x303)
        x305=self.conv2d48(x304)
        x306=self.batchnorm2d48(x305)
        return x306

m = M().eval()
x302 = torch.randn(torch.Size([1, 352, 7, 7]))
start = time.time()
output = m(x302)
end = time.time()
print(end-start)
