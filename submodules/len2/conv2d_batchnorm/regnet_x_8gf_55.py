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
        self.conv2d55 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x179):
        x180=self.conv2d55(x179)
        x181=self.batchnorm2d55(x180)
        return x181

m = M().eval()
x179 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x179)
end = time.time()
print(end-start)
