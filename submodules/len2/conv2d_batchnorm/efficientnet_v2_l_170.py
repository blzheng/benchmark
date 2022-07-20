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
        self.conv2d262 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d170 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x843):
        x844=self.conv2d262(x843)
        x845=self.batchnorm2d170(x844)
        return x845

m = M().eval()
x843 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x843)
end = time.time()
print(end-start)
