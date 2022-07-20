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
        self.conv2d99 = Conv2d(208, 1248, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d59 = BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x304, x289):
        x305=operator.add(x304, x289)
        x306=self.conv2d99(x305)
        x307=self.batchnorm2d59(x306)
        return x307

m = M().eval()
x304 = torch.randn(torch.Size([1, 208, 7, 7]))
x289 = torch.randn(torch.Size([1, 208, 7, 7]))
start = time.time()
output = m(x304, x289)
end = time.time()
print(end-start)
