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
        self.conv2d163 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d105 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x519):
        x520=self.conv2d163(x519)
        x521=self.batchnorm2d105(x520)
        return x521

m = M().eval()
x519 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x519)
end = time.time()
print(end-start)
