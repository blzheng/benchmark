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
        self.batchnorm2d53 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x347):
        x348=self.batchnorm2d53(x347)
        x349=self.conv2d54(x348)
        return x349

m = M().eval()
x347 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x347)
end = time.time()
print(end-start)
