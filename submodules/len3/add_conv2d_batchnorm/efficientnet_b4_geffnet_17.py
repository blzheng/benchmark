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
        self.conv2d119 = Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d71 = BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x352, x338):
        x353=operator.add(x352, x338)
        x354=self.conv2d119(x353)
        x355=self.batchnorm2d71(x354)
        return x355

m = M().eval()
x352 = torch.randn(torch.Size([1, 272, 7, 7]))
x338 = torch.randn(torch.Size([1, 272, 7, 7]))
start = time.time()
output = m(x352, x338)
end = time.time()
print(end-start)
