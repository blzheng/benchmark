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
        self.conv2d257 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d167 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x827):
        x828=self.conv2d257(x827)
        x829=self.batchnorm2d167(x828)
        return x829

m = M().eval()
x827 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x827)
end = time.time()
print(end-start)