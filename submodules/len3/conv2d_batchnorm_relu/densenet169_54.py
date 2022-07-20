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
        self.conv2d110 = Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d111 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu111 = ReLU(inplace=True)

    def forward(self, x393):
        x394=self.conv2d110(x393)
        x395=self.batchnorm2d111(x394)
        x396=self.relu111(x395)
        return x396

m = M().eval()
x393 = torch.randn(torch.Size([1, 736, 7, 7]))
start = time.time()
output = m(x393)
end = time.time()
print(end-start)
