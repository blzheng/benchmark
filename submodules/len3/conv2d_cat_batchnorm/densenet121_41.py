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
        self.conv2d93 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d96 = BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x333, x313, x320, x327, x341):
        x334=self.conv2d93(x333)
        x342=torch.cat([x313, x320, x327, x334, x341], 1)
        x343=self.batchnorm2d96(x342)
        return x343

m = M().eval()
x333 = torch.randn(torch.Size([1, 128, 7, 7]))
x313 = torch.randn(torch.Size([1, 512, 7, 7]))
x320 = torch.randn(torch.Size([1, 32, 7, 7]))
x327 = torch.randn(torch.Size([1, 32, 7, 7]))
x341 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x333, x313, x320, x327, x341)
end = time.time()
print(end-start)
